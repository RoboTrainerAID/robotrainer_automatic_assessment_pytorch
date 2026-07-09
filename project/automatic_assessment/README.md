# RoboTrainer Automatic Assessment — ML Pipeline

Machine-learning pipeline that estimates clinical motor-performance scores
(strength, speed, coordination test battery) from multimodal human-robot
interaction data recorded with the **RoboTrainer**, a force-controlled
robotic walker (IROS 2026 paper: *"Automatic Assessment of Motor Performance
with Disturbance Forces by a Force-Controlled Robotic Walker using Machine
Learning"*). Study: n = 28 able-bodied adults, 20 path scenarios each, with
haptic disturbance forces and inverted-steering challenges.

> New here (human or AI)? Read this file top to bottom, then
> `open_improvements.md` (living list of open problems, ranked) and
> `analysis.md` (the 2026-07 code audit that motivated the current design).

---

# How to start

## 1. RAW dataset generation (external, only when new recordings arrive)
1. Place all `.bag` files in one folder
2. With docker `robotrainer_docker_meldic/gait` (branch `gait`) generate `gait.bag` files
3. With docker `robotrainer_docker_meldic/bag_to_csv` (branch `bag_to_csv`):
   check `config.ini`, run `bag_to_numpy_dataset.py`
4. Result: `/data/raw/timeseries_numpy/U<user>/path_<id>/*.npy`
   — one file per signal, columns `[raw_ts, rel_ts, value]`.
   **No value imputation happens in this step** (an old paper note claiming
   KNN imputation was wrong).

## 2. FINAL dataset generation
```bash
python -m automatic_assessment.dataset.process_dataset
```
- Reads `/data/raw/timeseries_numpy`, writes `/data/raw/timeseries_numpy_processed`
  and `/data/raw/timeseries_features.csv` (85 statistical/frequency features per path).
- Configuration: `src/automatic_assessment/dataset/config.py`
  (channel lists, feature definitions, model channel groups, augmentation).
- Fails loudly if the data violates assumptions (missing series after
  imputation, NaN features) — a clean exit 0 means the dataset is valid.

## 3. TRAIN / TEST split (+ augmentation + tensor precompute)
```bash
python -m automatic_assessment.framework.data.dataset
```
- Splits by user: test users `[14, 19, 27]` (most "average" users by normative
  reference data), everyone else → train.
- Generates **synthetic augmentation clones for training users only**
  (up to `config.AUGMENTATION["max_ratio"]`, currently 5 — see below).
- Precomputes the grouped model tensors into `ts_tensors_cache.npz` per split
  (self-heals on load if config/data changed).
- Re-run this whenever `process_dataset` output, the channel groups, or the
  augmentation config changed. Takes a few minutes.

## 4. RUN training + evaluation
```bash
python -m automatic_assessment.framework.main_simple
```
Edit `main_simple.py` before running:
- `models_to_test` — list of model classes (see model list below)
- `targets_to_test` — target subsets (default: the 8 learnable targets)
- `augmentation_range` — augmentation ratios to sweep (0 = originals only;
  ratios switch instantly via dataset views, no regeneration)
- `ExperimentConfig(...)` — `epochs`, `hyperparameter_mode`
  (`'default'` = fixed params | `'optimize'` = Optuna search with `n_trials`),
  `early_stopping_patience`, `seed`

**Outcome:** one folder per experiment under `experiment_results/`
`YYYYMMDD_HHMMSS_<ModelName>/` containing (artifact schema v3):
- `config.yaml` — full run config, best params, epoch budget, shapes
- `metrics.yaml` — nested metrics: `val`/`test` × `scaled`/`unscaled`,
  per-target blocks keyed by target NAME, per-fold summaries
- `predictions.csv` — tidy: one row per (split, user, target) with
  `y_true`/`y_pred` in scaled AND original units
- `metrics_per_target.csv`, `learning_curve.csv`,
  `tuning_trials.csv` + `param_importances.csv` (optimize mode),
  `model_source.py` (snapshot)
- `plots/val/…` and `plots/test/…` (parity, residuals, per-user error) +
  `plots/learning_curve.png`, `plots/rmse_comparison.png`

## 5. Plots & cross-run comparisons
- Per-run plots are generated automatically at the end of each experiment
  (`reporting/visualization.py`; can be re-run on any result folder, also
  legacy pre-v3 folders).
- Cross-run analysis (e.g. augmentation-ratio sweeps, model comparisons):
  ```bash
  python -m automatic_assessment.framework.reporting.comparison
  ```
  Point `ExperimentComparison(<folder-with-runs>, split="val")` at a directory
  of run folders. Produces `comparison_results/` with `ratio_summary_*.png`
  (mean RMSE vs ratio + per-target spaghetti + selection-gap for optimize
  runs), `ratio_fold_distributions.png` (per-user fold losses per ratio),
  per-target bar charts, model comparisons, and tidy CSVs
  (`run_summaries.csv`, `fold_summaries.csv`, `<split>_aggregated_metrics.csv`).
- Verify the whole pipeline after changes: `python tests/test_pipeline_smoke.py`
  (synthetic, fast, must print `ALL SMOKE TESTS PASSED`).

---

# Data pipeline in detail

## Raw data
Per (user, path) and signal one `.npy` array `[raw_ts, rel_ts, value]`.
25 raw channels: user force x/y + torque z (200 Hz), robot velocity x/y/rot
(50 Hz), robot position x/y/θ, path deviation front/left/right,
disturbance force x/y, heart rate (1 Hz), HRV + PPI (~2.2 Hz),
left/right stride duration/length/stance/swing (~1 Hz events).
Plus per-path scalars (duration, cadence, stride counts) and `meta.json`.

## `dataset/process_dataset.py` — processing chain (in order)
1. **Load** (`timeseries_loader.py`) — file discovery, scalar normalization.
2. **Preprocess** (`timeseries_preprocessor.py`) — HRV column fix, PPG
   flattening, trimming of mechanical channels to motion start
   (velocity threshold), removal of invalid zero rows in HR/HRV/PPI.
   Relative timestamps are NOT shifted, so all channels of a path share
   one clock.
3. **Validate (pre)** (`timeseries_validation.py`) — missing/empty series,
   missing scalars, NaN/Inf values.
4. **Impute** (`timeseries_imputer.py`) — ONLY whole-missing/empty series:
   replaced by a single-point user-internal robust mean (44 cases in the
   current data); zero-disturbance series added for the 5 paths without
   disturbances. **No value-level imputation anywhere** — missingness is
   carried as masks instead (deliberate design; see Challenges).
5. **Validate (post)** — any remaining issue is a hard error.
6. **Derive** (`timeseries_derived_ts.py`) — force/velocity magnitudes
   (incl. the Eq.-4 total force with tangential torque component), power,
   cumulative work → 32 channels total.
7. **Extract features** (`timeseries_features.py`) — per path 85 statistical /
   frequency / correlation / time-delay features from the FULL-RESOLUTION
   series (trimmed mean, std, p05/p95, RMS, band powers 0.1–2 Hz and
   3–10 Hz, energy/impulse/work integrals, cross-correlations, lags).
   Feature keys are validated against the channel config (unknown keys raise).

## Split + augmentation (`framework/data/dataset.py`)
- `prepare_and_split_data` merges `timeseries_features.csv` +
  `task_difficulty.csv` + `demographics.csv` + `motoric_test.csv` (targets),
  splits by user, writes per-split CSVs + raw series.
- **Augmentation** (`dataset/timeseries_augmentation.py`, train split only):
  a clone is a *noisy re-measurement of the same person* — targets,
  demographics, task difficulty stay bit-identical. Per (path, channel)
  jitter with `σ = noise_factor × robust (IQR) std`; positive-only channels
  (physio, gait) get multiplicative truncated noise; constant traces stay
  constant; derived series re-derived and features re-extracted from the
  noisy traces. Clone id = `original*100 + index`, deterministic RNG per
  clone. Config: `config.AUGMENTATION`. Test users NEVER get clones.
- **Grouped tensor precompute**: the learned-feature models consume one
  tensor per channel group (`config.TS_MODEL_GROUPS`):

  | group | rate | channels | rationale |
  |---|---|---|---|
  | mechanical | 50 Hz | 21 | keeps force dynamics + tremor band (Nyquist 25 Hz); 200 Hz raw bin-averaged 4:1 |
  | physiological | 2 Hz | 3 | native HRV/PPI resolution; HR (1 Hz) has masked gaps |
  | gait | 2 Hz | 8 | ~1 event/stride, events stay separate |

  Within a group all channels share one time grid (bins = 1/rate, sample
  means, boolean validity mask); **groups are NOT aligned to each other**.
  Padding/missing = value 0 + mask False. Statistical features are never
  affected by this binning (they use the raw series).

## The X schema (`framework/data/schema.py` — single source of truth)
```
X = (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)
```
- `x_path (N, 20, 88)` — 85 extracted features + task-difficulty per path
- `x_user (N, 2)` — age, gender
- per group: `g_x / g_mask (N, 20, C_g, T_g)`
- `y (N, n_targets)` — clinical scores (up to 14; default subset = 8
  learnable ones)

Never hard-code tuple indices — use `schema.split_inputs`,
`schema.group_shapes`, `schema.iter_group_tensors`. **Validity always comes
from the masks, never from values** (padding is re-zeroed after scaling; the
old `|x|>1e-8` sentinel was a bug class).

`AssessmentDataset(folder)` loads a split (finite-checked);
`dataset.view(targets=[...], augmentation_ratio=r)` returns an immutable
`DatasetView` — datasets are stateless, views select.

---

# Machine-learning pipeline in detail

Entry: `framework/main_simple.py` → `framework/core/pipeline_simple.py`
(`SimplePipeline`). All results are typed dataclasses
(`framework/data/results.py`: `ExperimentConfig`, `MetricTable`,
`FoldResult`, `ValidationResult`, `TestResult`, `ExperimentResult`).

## Validation: Leave-One-Group-Out CV (the PRIMARY metric)
- `AugmentedLOGO` (`framework/data/data_utils.py`): one fold per ORIGINAL
  user; the held-out user's augmentation clones are excluded from that
  fold's training set; validation is always on original users only.
- Inside every fold (fitted on the fold-train only — no leakage):
  1. **Scaling**: `StandardScaler` for `x_path`, `x_user`, `y`; the group
     tensors get per-channel MASKED standardization (statistics over valid
     bins only, padding re-zeroed). NaN/Inf anywhere → hard error.
  2. **Feature selection**: LARS (`n_path_features`, default per model) or
     correlation threshold on the path features.
  3. **Training** (`framework/core/trainer.py`): Huber loss (δ=1) on scaled
     targets, AdamW + cosine annealing, batch 6. Early stopping (patience
     from config) monitors the fold's val loss and **restores the
     best-epoch weights**; the best epoch is recorded per fold.
- Per-user predictions (scaled + inverse-transformed to original units) and
  the dummy (train-mean) baseline — evaluated on the same folds/scalers —
  go into the `ValidationResult`.

## Hyperparameters
- `'default'` mode: each model's `get_default_parameters()`.
- `'optimize'` mode: Optuna TPE (seeded) over `get_hyperparameter_space()`,
  objective = mean LOGO-CV loss. Afterwards the winning config is
  **re-evaluated once** — `metrics.yaml` stores both `best_trial_val_loss`
  (selection score, optimistically biased: winner's curse over n_trials)
  and the re-evaluated metrics. Compare models only at equal `n_trials`.

## Final test (optional stage 2)
- One model trained on ALL training users for the **mean best-epoch from CV**
  (early stopping on the test evaluation itself is forbidden — it would let
  the test set pick the model), evaluated once on the 3-user hold-out.
- Test results are ADDED to a new `ExperimentResult`; tuning results are
  never mutated. Treat test numbers as a normative spot check (n=3!), not
  as a benchmark — validation is the decision metric.

## Reproducibility
`set_global_seed(seed)` (random/numpy/torch/CUDA) at the start of every
experiment + seeded Optuna sampler. Verified: identical runs are
bit-identical. Sklearn models are internally seeded (42).

## Available models (`framework/models/`)
Time-series models consume the grouped tensors (one encoder per group)
AND the path/user features; the others use path+user features only.

| Model class | file | uses ts | idea |
|---|---|---|---|
| `CNNBaselineNOEMBED(FLAT)` | timeseries/CNN_baseline_no_embedding(_flat).py | ✔ | per-group Conv1d, masked mean pooling, no bottleneck (current best) |
| `MultiScaleTCN` | timeseries/TCN_multiscale.py | ✔ | multi-scale dilated TCN encoder in the same simple dual-track topology as HybridCNNGRUFusion (controlled encoder A/B); v3 — the v1/v2 MIL-attention/coverage-gating variants scored 0.935/1.009 and were stripped (lesson log in the docstring) |
| `HybridCNNGRUFusion` | timeseries/CNN_hybrid_fusion.py | ✔ | combines the sweep winners — CNN_NOEMBED encoder + small GRU on downsampled features + BASE_FLAT compressed-flatten + mean dual aggregation, mean+std pooling; v2 val 0.8636 ≈ BASE_FLAT 0.8628, clearly better on the Robotrainer targets (0.68–0.78), worse on grip/weak-signal targets |
| `CNNBaseline` | timeseries/CNN_baseline.py | ✔ | as above + path bottleneck (default bottleneck is tight — see open_improvements) |
| `LSTMBaseline` | timeseries/LSTM_baseline.py | ✔ | per-group LSTM, packed to mask lengths (skips padding) |
| `HierarchicalTimeseriesLSTM` | timeseries/LSTM.py | ✔ | shared 1-channel LSTM over every channel (slow: Python channel loop) |
| `BASEBaseline / FLAT / NORM` | timeseries/BASE_baseline*.py | – | linear path encoder; FLAT flattens path embeddings (strong baseline) |
| `MLPBaseline`, `MLPSharedEncoder(FLAT)`, `MLPPathSpecific` | mlp_*.py | – | MLP variants over path features |
| `SimpleMLPRegressor`, `LinearRegressionModel`, `ElasticNetModel`, `RandomForestLikeMLP` | simple_models.py | – | torch baselines (ElasticNet penalty applied via trainer hook) |
| `LinearReg`, `ElasticNetReg`, `SVRReg`, `RandomForestReg`, `SGDReg`, `TabPFNReg`, `AutoSklearnReg` | sklearn/sklearn_models.py | – | sklearn baselines on flattened path+user features |

New model? Copy `models/template_model.py` — it documents the input
convention and mask helpers (`models/timeseries/masking.py`).

---

# Project context & knowledge

- **Goal**: replace manual clinical assessment with continuous automatic
  assessment during robotic walker training, eventually enabling adaptive
  difficulty. Current phase: architecture development on a pilot cohort of
  able-bodied adults; a future cohort will serve as the real test set.
- **Targets**: motor test battery (Bös et al. protocols). 6 of 14 targets
  (balance/coordination heavy) proved unlearnable on this cohort and are
  excluded by default (`best_performing_targets`, 8 targets).
- **Test set philosophy**: the 3 hold-out users were chosen as the most
  "average" users w.r.t. normative reference datasets (TOST-validated) —
  they answer "does it work on an average unseen user?", nothing more.
  Test often looks BETTER than validation for exactly this reason.
  **Validation (LOGO-CV) is the decision metric for all development.**
- **Milestone findings** (details: `open_improvements.md` §0/§3b):
  - Grouped full-resolution representation beats the old 1 Hz-averaged one:
    since its introduction the TS models (CNN_NOEMBED 0.852 val RMSE)
    lead the hand-crafted-features-only models (BASE_FLAT 0.863) for the
    first time. Dummy baseline: 1.089.
  - Augmentation ratio experiment (BASE_FLAT, default params): ratio 1
    gives a small significant gain (0.863→0.844, 17/25 users better,
    Wilcoxon p=0.03); more clones do not help; with per-ratio Optuna the
    effect drowns in selection noise.
- **History**: the pipeline was heavily audited and rebuilt in 2026-07
  (`analysis.md`): metric-argument bug, broken padding masks, val-as-test
  relabeling, unseeded runs, and a mis-keyed feature config were all fixed.
  Results older than 2026-07-06 are not comparable to current runs.
- **Paper deviations to know about**: the paper mentions KNN imputation
  (wrong — none exists anywhere, by design) and reports the biased
  best-trial validation scores of the pre-audit pipeline.

## Key documents
- `open_improvements.md` — LIVING document: ranked open problems (O#),
  refactoring proposals (R#), experiment findings, decisions log. Update it
  whenever problems are found or resolved.
- `analysis.md` — frozen 2026-07-03 audit report (the "why" behind many
  current design rules).
- `README_old.md` — the historical README (pre-rebuild), kept for reference.

---

# Challenges of this ML problem (read before designing anything)

1. **Tiny N**: 28 users total, 25 for training → LOGO-CV with 1-sample
   folds; every metric is noisy; seed variance rivals model differences.
   Always compare with per-fold pairing and/or multiple seeds.
2. **Selection bias everywhere**: hyperparameter search, model choice, and
   target pruning all select on the same 25 folds (winner's curse). Report
   re-evaluated scores; keep n_trials equal across models; the future
   cohort is the only untouched estimate.
3. **Wildly mixed sampling rates**: 200 Hz forces vs 1 Hz heart rate —
   solved by per-group tensors at group-appropriate rates; never average
   everything to one rate (hides the force dynamics the study is about).
4. **Sparse/irregular signals**: HRV/PPI cover only ~4% of the mechanical
   timeline; gait parameters are events; whole series can be missing
   (1-point imputed stubs). Masks — not imputed values — carry this
   information; models must pool/pack with masks.
5. **Noisy human data + hard concept**: clinical scores from interaction
   behavior is a weak-signal problem; the dummy baseline (RMSE 1.0 scaled)
   is embarrassingly competitive on some targets (esp. coordination).
6. **Test set is not a benchmark**: n=3, deliberately average, selected
   using target information. Never rank models by it, never early-stop on it.
7. **Pseudo-replication**: 20 paths per user (and clones at ratio>0) are
   not independent samples — feature selection and any statistics must
   respect user grouping.

---

# Notes for new contributors (human or AI)

- **Run the smoke test first and after every change**:
  `python tests/test_pipeline_smoke.py` — it checks the invariants that
  broke historically (metric argument order, mask-based scaling, early-stop
  weight restore, result immutability, artifact schema, view filtering).
- **Environment**: Docker with CUDA torch preinstalled; `pyproject.toml`
  deliberately does NOT list torch (see comment there). Data lives under
  `/data` (not in the repo).
- **Conventions that must not regress**:
  - metrics: always `y_true` first (`MetricTable.from_predictions`)
  - validity from masks, never from tensor values
  - anything fitted (scaler, selection, tuning) lives INSIDE the CV fold
  - validation results are never relabeled as test results
  - fail loudly on NaN/Inf and schema mismatches — no silent fallbacks
  - `archiv/` is frozen reference code and does not import — never revive
    it without migrating to the current schema
- **After changing** `dataset/config.py` (channels, groups, features,
  augmentation) or anything in `process_dataset`: re-run steps 2 + 3 above;
  the tensor cache self-heals, the split CSVs do not.
- **Where things live**: input layout → `framework/data/schema.py`;
  result objects → `framework/data/results.py`; fold logic →
  `framework/core/pipeline_simple.py`; training loop →
  `framework/core/trainer.py`; per-run artifacts → `reporting/saving.py`
  (schema v3) and `reporting/visualization.py`; cross-run analysis →
  `reporting/comparison.py` (v3 + legacy folders).
- **Open work**: see `open_improvements.md` — top items are seed-robustness
  studies of the model ranking, physiological-channel strategy, and the
  remaining reporting-script ports (`paper_comparison*.py`,
  `hyperparameter_analysis.py` still expect the legacy schema).
