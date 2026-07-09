"""
Synthetic end-to-end smoke test for the SimplePipeline framework.

Runs entirely on random data (no real dataset needed, tiny and fast) and
asserts the invariants that were broken before the 2026-07 fixes:

1. Masked scaling: statistics use only valid entries; padding stays 0.
2. Mask utilities: lengths / masked means derived from the mask.
3. calculate_metrics argument order: val R^2 equals a manual
   r2_score(actuals, preds) computation (analysis 3.1).
4. Early stopping restores best weights and reports best_epoch (2.3).
5. Tuning results contain only val_* keys (no fake test relabeling, 3.2),
   final test ADDS test_* keys without mutating tuning results (3.10).
6. Saving + visualization work for tuning-only AND tuning+test results.
7. ElasticNetModel's elastic-net penalty is actually applied (3.7).

Run:  python tests/test_pipeline_smoke.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.metrics import r2_score

from automatic_assessment.framework.utils.seed import set_global_seed
from automatic_assessment.framework.data.data_utils import (
    prepare_fold_data, _masked_channel_scaling
)
from automatic_assessment.framework.data.results import ExperimentConfig
from automatic_assessment.framework.models.timeseries.masking import (
    timestep_mask, masked_mean_over_time, mask_lengths
)
from automatic_assessment.framework.core.pipeline_simple import SimplePipeline
from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.models.timeseries.CNN_baseline import CNNBaseline
from automatic_assessment.framework.models.simple_models import ElasticNetModel
from automatic_assessment.framework.models.sklearn.sklearn_models import LinearReg
from automatic_assessment.framework.reporting.saving import SavingModule
from automatic_assessment.framework.reporting.visualization import VisualizationModule


def _make_group(rng, n_users, n_paths, n_channels, T):
    """One synthetic channel-group tensor pair with ragged lengths and gaps."""
    x = np.zeros((n_users, n_paths, n_channels, T), dtype=np.float32)
    mask = np.zeros((n_users, n_paths, n_channels, T), dtype=bool)
    for u in range(n_users):
        for p in range(n_paths):
            length = rng.integers(max(3, T // 3), T + 1)
            for c in range(n_channels):
                valid = np.zeros(T, dtype=bool)
                valid[:length] = True
                if c == 0 and length > 3:
                    valid[2] = False  # internal gap
                x[u, p, c, valid] = rng.normal(loc=2.0, scale=1.0, size=valid.sum())
                mask[u, p, c] = valid
    return torch.tensor(x), torch.tensor(mask)


def make_synthetic_dataset(n_users=8, n_paths=3, f_path=6, f_user=2, n_targets=3):
    """
    Random dataset following the grouped convention:
        X = (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask)
    Two groups with different channel counts and DIFFERENT lengths
    (mimicking mechanical @ high rate vs physiological @ low rate).
    """
    rng = np.random.default_rng(0)

    g0_x, g0_mask = _make_group(rng, n_users, n_paths, n_channels=4, T=40)
    g1_x, g1_mask = _make_group(rng, n_users, n_paths, n_channels=2, T=10)

    x_path = rng.normal(size=(n_users, n_paths, f_path)).astype(np.float32)
    x_user = rng.normal(size=(n_users, f_user)).astype(np.float32)

    # Targets correlated with the path features so the models can learn
    w = rng.normal(size=(f_path, n_targets))
    y = (x_path.mean(axis=1) @ w + 0.1 * rng.normal(size=(n_users, n_targets))).astype(np.float32)

    X = (torch.tensor(x_path), torch.tensor(x_user), g0_x, g0_mask, g1_x, g1_mask)
    users = np.arange(1, n_users + 1)
    return X, torch.tensor(y), users


def test_masked_scaling():
    X, y, users = make_synthetic_dataset()
    # Layout: (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask)
    g0_x, g0_mask = X[2], X[3]

    mean, std = _masked_channel_scaling(g0_x, g0_mask)
    # Manual masked statistics for channel 1
    c = 1
    vals = g0_x[:, :, c, :][g0_mask[:, :, c, :]]
    assert abs(mean[c].item() - vals.mean().item()) < 1e-4, "masked mean wrong"
    assert abs(std[c].item() - vals.std(unbiased=False).item()) < 1e-3, "masked std wrong"

    # Full fold preparation: padding must stay exactly 0 after scaling,
    # for EVERY group
    Xt_s, yt_s, Xv_s, yv_s, scaler_y, _ = prepare_fold_data(
        tuple(t[:6] for t in X), y[:6], tuple(t[6:] for t in X), y[6:]
    )
    for gi in range(2, len(Xt_s), 2):
        assert torch.all(Xt_s[gi][~Xt_s[gi + 1]] == 0.0), f"padding not re-zeroed (train, group {gi})"
        assert torch.all(Xv_s[gi][~Xv_s[gi + 1]] == 0.0), f"padding not re-zeroed (val, group {gi})"
        valid_vals = Xt_s[gi][Xt_s[gi + 1]]
        assert abs(valid_vals.mean().item()) < 0.15, f"valid entries not standardized (group {gi})"
    print("PASS masked scaling")


def test_mask_utils():
    mask = torch.tensor([[1, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    lengths = mask_lengths(mask)
    assert lengths.tolist() == [4, 1, 6], f"mask_lengths wrong: {lengths.tolist()}"

    feat = torch.ones(3, 6, 2)
    pooled = masked_mean_over_time(feat, mask.float())
    assert torch.allclose(pooled[0], torch.ones(2)), "masked mean should ignore invalid steps"
    assert torch.allclose(pooled[1], torch.zeros(2)), "empty sequence should pool to zeros"

    x_mask4 = mask.view(3, 1, 1, 6).expand(3, 1, 2, 6)
    t_mask = timestep_mask(x_mask4)
    assert t_mask.shape == (3, 1, 6)
    print("PASS mask utils")


def test_early_stopping_restores_best_weights():
    set_global_seed(0)
    X, y, users = make_synthetic_dataset()
    Xt = tuple(t[:6] for t in X)
    Xv = tuple(t[6:] for t in X)
    Xt_s, yt_s, Xv_s, yv_s, _, _ = prepare_fold_data(Xt, y[:6], Xv, y[6:])

    params = CNNBaseline.get_default_parameters()
    params["n_path_features"] = None  # skip LARS for this unit test
    input_dims = CNNBaseline.get_input_dims(Xt_s)
    trainer = Trainer(CNNBaseline, input_dims, yt_s.shape[1], params)

    history = trainer.train_model_and_evaluate_every_epoch(
        Xt_s, yt_s, epochs=6, X_val=Xv_s, y_val=yv_s, early_stopping_patience=2
    )
    best_epoch = trainer.last_best_epoch
    assert best_epoch is not None and best_epoch >= 1
    # The evaluated loss must equal the BEST epoch's val loss (weights restored)
    val_loss, _, _ = trainer.evaluate_model(Xv_s, yv_s)
    best_hist = min(history['val_loss'])
    assert abs(val_loss - best_hist) < 1e-5, (
        f"early stopping did not restore best weights: eval {val_loss} vs best {best_hist}"
    )
    trainer.cleanup()
    print(f"PASS early stopping (best_epoch={best_epoch}, restored loss={val_loss:.4f})")


def test_jitter_augmentation():
    from automatic_assessment.dataset.timeseries_augmentation import jitter_series, robust_std

    x = (np.sin(np.linspace(0, 10, 500)) * 3 + 5).astype(np.float64)

    out = jitter_series(x, 0.05, positive=False, rng=np.random.default_rng(1))
    assert out.shape == x.shape and not np.allclose(out, x)
    resid_std = float(np.std(out - x))
    rs = robust_std(x)
    assert 0.02 * rs < resid_std < 0.10 * rs, f"noise scale off: {resid_std} vs robust std {rs}"

    # Constant traces (e.g. zero-disturbance series) stay exactly constant
    const = np.full(100, 3.0)
    assert np.array_equal(jitter_series(const, 0.05, False, np.random.default_rng(2)), const)

    # Positive channels stay strictly positive under multiplicative noise
    pos = np.abs(np.random.default_rng(0).normal(1.0, 0.5, 300)) + 0.05
    outp = jitter_series(pos, 0.05, positive=True, rng=np.random.default_rng(3))
    assert (outp > 0).all(), "positive channel produced non-positive values"

    # Deterministic given the same RNG stream
    a = jitter_series(x, 0.05, False, np.random.default_rng(7))
    b = jitter_series(x, 0.05, False, np.random.default_rng(7))
    assert np.array_equal(a, b)
    print("PASS jitter augmentation")


def test_dataset_view():
    from automatic_assessment.framework.data.dataset import make_view
    from automatic_assessment.framework.data.schema import TSGroup

    X, y, _ = make_synthetic_dataset(n_users=6)
    # users: 3 originals + clones (id = orig*100 + clone_idx)
    users = np.array([1, 2, 3, 101, 102, 201])
    groups = [
        TSGroup("g0", 8.0, ["a", "b", "c", "d"], X[2], X[3]),
        TSGroup("g1", 2.0, ["e", "f"], X[4], X[5]),
    ]
    fn = {"targets": ["T0", "T1", "T2"], "path": [], "user": [], "ts_groups": []}

    v0 = make_view(X[0], X[1], groups, y, users, fn, targets=["T2", "T0"], augmentation_ratio=0)
    assert list(v0.users) == [1, 2, 3], "ratio=0 must keep originals only"
    assert v0.feature_names["targets"] == ["T2", "T0"]
    assert torch.allclose(v0.y, y[:3][:, [2, 0]]), "target reordering wrong"
    assert v0.ts_groups[0].x.shape[0] == 3

    v1 = make_view(X[0], X[1], groups, y, users, fn, augmentation_ratio=1)
    assert list(v1.users) == [1, 2, 3, 101, 201], "ratio=1 must add clone index 1 only"
    v2 = make_view(X[0], X[1], groups, y, users, fn, augmentation_ratio=2)
    assert list(v2.users) == [1, 2, 3, 101, 102, 201]

    # Parent must stay untouched (stateless views, R11)
    assert y.shape[1] == 3 and len(users) == 6

    try:
        make_view(X[0], X[1], groups, y, users, fn, targets=["nope"])
        assert False, "unknown target must raise"
    except ValueError:
        pass
    print("PASS dataset view (R11)")


def test_elasticnet_penalty_applied():
    set_global_seed(0)
    X, y, users = make_synthetic_dataset()
    input_dims = ElasticNetModel.get_input_dims(X)
    params = ElasticNetModel.get_default_parameters()
    params["alpha"] = 1000.0  # extreme penalty must dominate the loss
    model = ElasticNetModel(input_dims, y.shape[1], params)
    reg = model.regularization_loss()
    assert reg.item() > 1.0, "elastic net penalty not applied"
    print(f"PASS elastic net penalty (reg={reg.item():.1f})")


def run_pipeline(model_class, X, y, users, tmpdir):
    targets = [f"T{i}" for i in range(y.shape[1])]
    config = ExperimentConfig(
        epochs=3,
        hyperparameter_mode="default",
        early_stopping_patience=2,
        seed=0,
        targets=targets,
        note="smoke test",
    )
    set_global_seed(0)
    pipeline = SimplePipeline(model_class, config)
    tuning = pipeline.run_simple_tuning(X, y, users)

    # ---- Typed result checks: tuning results must be val-only ----
    assert tuning.stage == "tuning"
    assert tuning.test is None, "tuning results must not contain test results"
    assert len(tuning.validation.folds) == len(users), "one CV fold per user expected"
    assert tuning.validation.metrics.targets == targets

    # ---- Metric orientation check (3.1) ----
    actuals = np.concatenate([f.val_actuals for f in tuning.validation.folds])
    preds = np.concatenate([f.val_preds for f in tuning.validation.folds])
    manual_r2 = r2_score(actuals, preds)
    saved_r2 = tuning.validation.metrics.r2_mean
    assert abs(manual_r2 - saved_r2) < 1e-5, (
        f"val R2 mis-ordered: manual {manual_r2:.4f} vs saved {saved_r2:.4f}"
    )

    # ---- Baseline evaluated on same folds, incl. unscaled ----
    assert tuning.validation.baseline_metrics.rmse_mean > 0
    assert tuning.validation.baseline_metrics_unscaled.rmse_mean > 0

    # ---- Save + visualize tuning-only results (schema v3) ----
    saver = SavingModule(model_name=f"{model_class.model_name}_tuneonly", output_root=tmpdir)
    saver.save_results(tuning)
    assert os.path.exists(os.path.join(saver.output_dir, "predictions.csv"))

    import yaml
    with open(os.path.join(saver.output_dir, "metrics.yaml")) as f:
        m = yaml.safe_load(f)
    assert m["schema_version"] == 3
    assert m["stage"] == "tuning"
    assert "rmse_mean" in m["val"]["scaled"]
    assert set(m["val"]["scaled"]["per_target"].keys()) == set(targets)
    assert "test" not in m

    viz = VisualizationModule(saver.output_dir)
    viz.generate_all_plots()
    assert os.path.exists(os.path.join(saver.output_dir, "plots", "val", "parity_plots.png"))

    # ---- Final test on a synthetic hold-out ----
    X_test, y_test, users_test = make_synthetic_dataset(n_users=3)
    final = pipeline.run_final_test(X, y, X_test, y_test, users_test, tuning)

    assert final.stage == "tuning+final_test"
    assert final.test is not None and final.test.metrics.rmse_mean > 0
    # tuning result object must NOT have been mutated
    assert tuning.test is None and tuning.stage == "tuning"

    saver2 = SavingModule(model_name=f"{model_class.model_name}_full", output_root=tmpdir)
    saver2.save_results(final)
    viz2 = VisualizationModule(saver2.output_dir)
    viz2.generate_all_plots()
    for split in ("val", "test"):
        for name in ("parity_plots.png", "residuals.png", "user_performance.png"):
            path = os.path.join(saver2.output_dir, "plots", split, name)
            assert os.path.exists(path), f"missing plot: {path}"

    # ---- Tidy predictions.csv: one row per (split, user, target) ----
    import pandas as pd
    preds_df = pd.read_csv(os.path.join(saver2.output_dir, "predictions.csv"))
    assert {"split", "fold", "user_id", "target",
            "y_true", "y_pred", "y_true_unscaled", "y_pred_unscaled"} <= set(preds_df.columns)
    assert set(preds_df["split"].unique()) == {"val", "test"}
    assert (preds_df["split"] == "val").sum() == len(users) * len(targets)
    assert (preds_df["split"] == "test").sum() == len(users_test) * len(targets)

    pt = pd.read_csv(os.path.join(saver2.output_dir, "metrics_per_target.csv"))
    assert set(pt["split"].unique()) == {"val", "test"}
    assert set(pt["scale"].unique()) == {"scaled", "unscaled"}

    print(f"PASS pipeline end-to-end for {model_class.model_name} "
          f"(val RMSE {tuning.validation.metrics.rmse_mean:.3f}, "
          f"final epochs {tuning.final_epochs})")


def main():
    tmpdir = tempfile.mkdtemp(prefix="aa_smoke_")
    try:
        test_mask_utils()
        test_masked_scaling()
        test_jitter_augmentation()
        test_dataset_view()
        test_early_stopping_restores_best_weights()
        test_elasticnet_penalty_applied()

        X, y, users = make_synthetic_dataset()
        run_pipeline(CNNBaseline, X, y, users, tmpdir)   # exercises mask-based pooling + torch trainer
        run_pipeline(LinearReg, X, y, users, tmpdir)     # exercises sklearn trainer path

        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
