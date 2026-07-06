# Codebase & Methodology Analysis Report

**Scope.** Read-only static analysis of `src/automatic_assessment/` (canonical pipeline: `framework/main_simple.py` → `SimplePipeline`; dataset generation: `dataset/`), the legacy pipelines (`framework/main.py`+`pipeline.py`, `sklearn_test/`, `archiv/`, `one_shots/`), the saved run artifacts in `experiment_results/`, `pyproject.toml`/`poetry.lock`, and the accompanying IROS paper (Zachariae et al. 2026). Every claim below cites `file:line` in this repository. No fixes were applied.

---

## 1. Executive Summary

The repository implements the paper's machine-learning pipeline: a padded tensor dataset `(X_ts, X_path, X_user) → Y`, Leave-One-Group-Out CV over users, Optuna hyperparameter search, LARS-based feature selection, and a family of PyTorch/sklearn regressors evaluated against a 3-user normative hold-out test set (IDs 14, 19, 27 — matching the paper).

Several important things are done **right**, and better than in many small-N studies: input and target scalers are fit **inside** each CV fold (`framework/data/data_utils.py:88-151`), LARS/correlation feature selection is fit on the fold-training data only (`framework/core/pipeline_simple.py:325-327`), group integrity is respected everywhere via a custom splitter that even anticipates augmented-clone leakage (`data_utils.py:163-203`), test-set metrics are inverse-transformed to physical units (`pipeline_simple.py:205-212`), and each run snapshots its model source and config (`framework/reporting/saving.py:15-23`).

However, the audit found **one outright metric-computation bug that is empirically visible in the saved artifacts** (swapped `y_true`/`y_pred` in all validation-metric calls → garbage R² values, e.g. `test_r2_target_1: -156.1` in `experiment_results/20260217_100712_SimpleMLPRegressor/metrics.yaml:85`), **one architectural defect that silently disables the deep models' core mechanism** (StandardScaler destroys the zero-padding sentinel that the CNN/LSTM masking depends on), and a cluster of methodological problems that matter greatly at N=25 training users: non-nested hyperparameter selection whose tuning objective is reported as the validation result, early stopping driven by the very fold being scored, a "validation-as-test" relabeling path that produces misleading saved artifacts, target-set pruning based on the same CV scores that are later reported, and a complete absence of random-seed control on the PyTorch side. Additionally, the imputation described in the paper (fold-wise KNN, k=1) **does not exist anywhere in this repository**, and the declared environment cannot even run the code (`torch` is absent from `pyproject.toml` and `poetry.lock`).

None of these issues suggests intent; the code is honest about some of them (the docstring itself calls the tuning estimate "optimistic/biased", `pipeline_simple.py:32-34`). But collectively they mean: (a) the published *validation* numbers are upward-biased model-selection scores, not generalization estimates; (b) the CNN/LSTM results do not measure what the paper says they measure (masked learned temporal features); (c) exact reproduction of any number in the paper from this repository is currently impossible. The hold-out *test* RMSEs are the most defensible numbers, but they rest on n=3 users and a pipeline-induced train/test distribution shift (per-split padding lengths).

Test coverage is zero (`tests/` contains a single empty `__init__.py`), which is how most of the silent failures below survived.

---

## 2. Scientific & Methodological Flaws

### 2.1 Non-nested tuning: the optimization objective is reported as the validation result — CRITICAL

`run_simple_tuning` ("Strategy B", `framework/core/pipeline_simple.py:28-152`) runs Optuna for up to `n_trials` (30–50), where each trial's objective is the mean LOGO-CV loss over **all** training users (`pipeline_simple.py:278-286`). The best trial's CV metrics are then returned and saved as the run's validation scores (`pipeline_simple.py:288-296`), and the paper's Table II ranks all nine models by exactly this metric (`RMSE_val`), as does the plotting utility (`framework/reporting/paper_comparison_plot_val.py:125`: "best model = lowest `mean_rmse_val`").

This is the winner's-curse configuration: with 30–50 trials evaluated on a noisy 25-fold CV (one user per fold), the *minimum* trial score is systematically lower than the true expected CV score of the chosen configuration. The docstring acknowledges the bias (`pipeline_simple.py:32-34`) but the reporting pipeline does not: nothing downstream distinguishes "tuning objective of the selected trial" from "validation performance". There is no outer loop around the search. The nested implementation exists (`framework/core/pipeline.py:22`, `run_nested_cv`) but is wired only to the archived legacy dataset (`framework/main.py:23,27`) and has its own leaks (§2.2).

Compounding selection effects on the same folds:
- **Target pruning**: the final 8-target set (`best_performing_targets`, `framework/main_simple.py:59-71`) was chosen because the other 6 targets validated at RMSE ≥ 1.0 (paper §V-A) — i.e., targets were selected on the same CV whose scores are then reported for the survivors.
- **Model-class selection**: nine models compared on the same folds; the reported best model inherits the same multiplicity bias.

At N=25 with per-fold noise this easily accounts for several hundredths of scaled RMSE — the same order as the gaps between models in Table II (0.84 vs 0.97).

### 2.2 The legacy "nested" CV is not leak-free either — HIGH (legacy path)

In `framework/core/pipeline.py`:
- The outer fold scales once on the outer-train set (`pipeline.py:71`), and the **inner** tuning CV then runs with `scale_data=False` (`pipeline.py:83,93,221` → `_evaluate_params_cv` default), i.e. every inner validation user was already included in the scaler fit. A small but strict inner-loop scaling leak — precisely the thing the fold-wise design elsewhere tries to prevent.
- `optimize_once` mode (`pipeline.py:85-93`) tunes hyperparameters on outer-fold-0's training data and **reuses them for all remaining outer folds**. Every later test user participated in the tuning data of fold 0, so the "unbiased" outer estimate is contaminated for 24 of 25 folds.
- Fold 0 passes the **outer test fold** as the per-epoch monitoring set (`pipeline.py:112`). No early stopping is active there (patience defaults to `None`), so no decision leaks, but test-fold curves are exposed to the experimenter mid-run.
- `only_first_fold: True` is the configured default in the legacy main (`framework/main.py:87`), silently turning "nested CV" into a single-fold estimate.

### 2.3 Early stopping is driven by the scoring fold, with no weight restoration — CRITICAL

In the canonical pipeline, each CV fold trains with `early_stopping_patience=2` monitoring the **same validation fold that produces the reported score** (`pipeline_simple.py:332-340` → `framework/core/trainer.py:80-108`; config `framework/main_simple.py:94`). Consequences:

1. **Optimistic bias**: the stopping epoch is chosen per-fold to (approximately) minimize the reported fold loss. This is a training decision conditioned on validation data — a textbook subtle leak, and it stacks on top of §2.1.
2. **The monitored quantity is a single sample**: with one sample per user and LOGO, each val fold has exactly 1 sample; a patience-2 rule on a 1-sample Huber loss is near-random epoch selection, injecting variance rather than regularization.
3. **No `best_weights` restore**: on patience expiry the loop just `break`s (`trainer.py:99-106`), so the model evaluated afterwards (`pipeline_simple.py:340`) carries the *last* weights — by construction 2 consecutive non-improving epochs past the best.
4. **Protocol mismatch with the final model**: `run_final_test` trains with `train_model` — full 30 epochs, no early stopping (`pipeline_simple.py:183`, `trainer.py:36-78`) — while the hyperparameters were selected under the early-stopped regime, and the CosineAnnealing schedule (`T_max=epochs`) only ever completes in the final run. The model whose test score is reported is trained under a different protocol than the models that justified its configuration. It also contradicts the paper's "trained for 30 epochs" description.

### 2.4 Zero-padding + pre-model standardization silently disables the CNN/LSTM masking — CRITICAL

The ragged time series (50–300+ steps) are packed into a dense tensor padded with `0.0`, with **no mask or length channel stored** (`framework/data/dataset.py:196-221`). All time-series models then infer the true length from the data itself via the zero-sentinel: `compute_lengths_flat` treats `|x| > 1e-8` as valid (`framework/models/timeseries/CNN_baseline.py:16-20`, `LSTM_baseline.py:20-27`, `BASE_baseline_flat.py:17-21`, and `LSTM.py:106-110`, which even uses `mask.sum()` — count of nonzeros — as "length").

But `prepare_fold_data` standardizes `x_ts` per channel over **all** timesteps — including the padding zeros — *before* the tensors reach any model (`data_utils.py:102-122`). After z-scoring, a padded zero becomes the constant `−μ/σ ≠ 0`. Therefore:

- **The masks are no-ops**: every sequence appears full-length; `masked_mean` averages over the entire padded window, and the LSTM's `pack_padded_sequence` receives full lengths and consumes the padded tail as real input (`LSTM_baseline.py:91-124`).
- **Signal dilution proportional to padding**: a 50-step path in a 300-step tensor contributes ~1/6 signal and ~5/6 constant; the path embedding largely encodes *sequence length / task duration*, not dynamics. The saved config of an earlier run shows the extreme case — `shape_X_ts: [25, 20, 32, 15573]` (`experiment_results/20260217_100712_SimpleMLPRegressor/config.yaml:5-9`), i.e. >99% padding for 1 Hz channels.
- **Scaler contamination**: μ and σ per channel are computed over the padding, so the amount of padding (an artifact of the longest path in the split) shifts every channel's scaling.
- **Pipeline-induced train/test shift**: `max_time` is computed independently per split (`dataset.py:183-187`), so `/data/train` and `/data/test` tensors have different lengths and different padding fractions; the pooled representation of a test user is systematically offset from training users by construction.

The paper's central claim for the deep branch — CNNs "take the raw time series as input … to automatically extract high-dimensional latent features" — is not what the shipped code computes. The published CNN/LSTM comparison is a comparison of length-confounded mean-pooled channel statistics.

### 2.5 The x_ts tensor ignores time: channels are positionally misaligned — HIGH

`_load_dataset_split` copies only the **value column**, writing sample *i* of every channel to tensor index *i* (`dataset.py:210-221`), discarding timestamps entirely. Upstream, however:
- Only mechanical signals are trimmed to motion start (`dataset/config.py:57-76`); `heart_rate`, `hrv`, `ppi` and stride parameters retain pre-motion samples, so their index 0 is a different wall-clock time than a trimmed channel's index 0 (`dataset/timeseries_preprocessor.py:75-119`).
- Zero-valued HR/HRV/PPI rows are **deleted**, not marked (`timeseries_preprocessor.py:121-127`), compacting those series and shifting all subsequent indices ("closest-next-value" semantics across gaps).
- Channels have heterogeneous sampling rates.

A `Conv1d` across the channel dimension (`CNN_baseline.py:44-48`) assumes cross-channel simultaneity at each index; that assumption is violated for essentially every physiological channel. The rule-based feature extractor does this correctly (interpolating onto a shared time base, `dataset/timeseries_features.py:276-310`) — the tensor path does not. Missing channels are silently all-zero (`dataset.py:210-221`), and the channel list itself is taken from the "first valid entry" per split with no train/test consistency check (`dataset.py:174-194`) — a silent feature-misalignment hazard.

### 2.6 Imputation: the paper's method does not exist; the implemented one is blunt — HIGH

- **Paper vs. code**: the paper states "Within each LOGOCV fold, missing values are imputed using KNN (k=1)". There is **no KNN imputer anywhere in this repository** (verified by search), and `prepare_fold_data` performs scaling only — no imputation of any kind happens inside the CV (`data_utils.py:88-151`).
- **What actually happens** (dataset generation, before any split — `dataset/process_dataset.py:30-63`): a whole *missing/empty* series is replaced by a **single datapoint** carrying the user's robust mean over their other paths (`dataset/timeseries_imputer.py:141-192`). This is user-internal (no cross-user leakage — good), but a 1-point series then cascades: `std=0`, percentiles collapse to the value, `rms_diff = mean of empty slice → NaN` (`timeseries_features.py:170-173`), integrals forced to 0.0 (`timeseries_features.py:133-168`), and the imputed timestamp is always 0.0 because `start_time` is never populated in that function (`timeseries_imputer.py:171-186`). It also erases the path-difficulty signal the whole assessment concept depends on.
- **Within-series NaNs/gaps (README: >10% for HRV/PPI) are never imputed at all** in the current pipeline — they survive as deleted rows (compaction, §2.5) or as NaN features (§3.5).
- **Legacy pipeline** (`sklearn_test/`): per-second resampling then `bfill().ffill()` per (user, path) (`sklearn_test/datasets.py:184-188`) — backward-fill copies **future** values into earlier timesteps, exactly the temporal leakage pattern flagged in the review brief. Group-wise CV keeps this from leaking across users, but every sample encodes future information, which misrepresents any real-time deployment claim. The `SmartImputer`'s `global_mean` strategy (`sklearn_test/imputers.py:51-54`) is the "global mean bias" the README itself criticizes; at least it is fitted inside the sklearn `Pipeline` within CV (`sklearn_test/models.py:74-93`).

### 2.7 Scaling anomalies — MEDIUM

- **Per-fold target scalers, concatenated metrics**: each LOGO fold fits its own `scaler_y` (`data_utils.py:90-97`), and predictions from 25 *different* scaled spaces are concatenated before computing RMSE/R² (`pipeline_simple.py:342-360`). With nearly identical folds the effect is small, but formally the "scaled" metric mixes 25 unit systems, and pooled-R² on LOO predictions is a known biased construct. The unscaled metrics (per-fold inverse transform, `pipeline_simple.py:346-349`) are the defensible ones.
- **Unbounded predictions**: the regression heads are unconstrained linear outputs; nothing clips predictions to the physical range of a clinical scale after inverse transform (the README itself raises this open question, `README.md:85`). For bounded scales this guarantees occasional impossible predictions.
- **Baseline is scaled correctly** (`pipeline_simple.py:363-397`), and `run_final_test` fits scalers on the full train set and only transforms the test set (`pipeline_simple.py:171`) — correct.

### 2.8 Statistical validity of the feature selection — MEDIUM

`select_multitarget_top_features_lars` and the correlation selector flatten `(N, P, F) → (N·P, F)` and `np.repeat` each user's target 20× (`framework/dimred/lars.py:154-157`, `framework/dimred/correlation.py:27-33`). LARS then sees ~480 "independent" samples that are actually 24 users; the regularization path and any implicit inference are computed under pseudo-replication. It is fold-internal (no leakage), but the selection is statistically overconfident and dominated by within-user redundancy. LARS failures are swallowed silently (`lars.py:195-196`), so a NaN-poisoned fold degrades to "no selection for that target" without any log.

### 2.9 Reproducibility: PyTorch and Optuna are completely unseeded — HIGH

There is no `torch.manual_seed`, `np.random.seed`, or seeded Optuna sampler anywhere in `framework/` (seeds exist only in the abandoned `one_shots/` scripts, e.g. `one_shots/experimental_pytorch.py:23-28`). Consequently: network initialization, dropout, GPU shuffling (`trainer.py:51`), DataLoader shuffling (`data_utils.py:66-82`), and the TPE search itself (`pipeline_simple.py:275`) are all non-deterministic. Sklearn models, by contrast, are pinned to `random_state=42` (`framework/models/sklearn/sklearn_models.py:50,138,201,283,339`) — so the published cross-family comparison mixes deterministic and irreproducible results. With N=25, run-to-run variance of a single unseeded training is easily the size of the model gaps in Table II.

### 2.10 Test-set design — MEDIUM (acknowledged in the paper, but compounded by the code)

The 3-user normative test set is hard-coded (`dataset.py:336-341`, IDs 14/19/27) and was selected using the users' **ground-truth motor scores** (paper §IV-C). This is defensible as a representativeness device, but (a) it is still information from the targets flowing into the evaluation design; (b) n=3 gives per-target test RMSEs that are essentially anecdotes (the paper's own "Max F Left" OOD example); (c) the val→test relabeling path (§3.2) means some saved artifacts present CV scores as "test" results.

### 2.11 Additional paper ↔ code discrepancies — HIGH (for reproducibility of the publication)

| Paper claim | Repository reality |
|---|---|
| "KNN (k=1) imputation within each LOGOCV fold" | No KNN imputer exists; no fold-wise imputation at all (§2.6) |
| "Path aggregation strategy (mean, learned weights, attention) was optimized" | All current search spaces hard-restrict to a single option — `["flatten"]` or `["mean"]`; `static`/`attention` are commented out (`BASE_baseline_flat.py:122`, `CNN_baseline.py:128`, `LSTM_baseline.py:160`, `LSTM.py:207-209`) |
| Force-magnitude features per Eq. 4 (with torque term) | `TS_FEATURES` keys features under `"user_force_mag"`, a series name that is never produced (derived name is `user_force_total_mag`, `dataset/config.py:183` vs `config.py:107-112`); the extractor silently skips unknown keys (`timeseries_features.py:95-96`), so **total-force features including the torque component were never extracted** — only `user_force_lin_mag` and `user_force_y` |
| "Models trained for 30 epochs" | Canonical config applies `early_stopping_patience=2` on 1-sample folds (§2.3); most folds train far fewer than 30 epochs |
| Augmentation (README step 2: "imputation, augmentation and merging") | `create_augmented_dataset` is an empty placeholder (`dataset.py:273-280`); `process_dataset.py` contains no augmentation step. Real augmentation exists only in `archiv/augmentation.py` (legacy). The `AugmentedLOGO` splitter is inert on the current data. `augmentation_ratio` recorded in configs is a no-op knob |
| 25 time-series channels (Xts ∈ ℝ^…×25×79) | `config.TIMESERIES_TO_LOAD` + derived = up to 34 channels; saved run shows 32 (`config.yaml:8`); the effective count depends on which channels the split's "first valid entry" happens to contain (§2.5) |

---

## 3. Logic Errors & Breaking Bugs

### 3.1 `calculate_metrics` called with swapped arguments — all scaled validation R² values are wrong — CRITICAL, empirically confirmed

Signature: `calculate_metrics(y_true, y_pred, prefix)` (`framework/reporting/metrics.py:5-7`). Two call sites pass `(predictions, actuals)`:

- `framework/core/pipeline_simple.py:354` — `calculate_metrics(all_val_preds, all_val_actuals, prefix="val")`
- `framework/core/pipeline.py:285` — same pattern in the nested pipeline

RMSE/MAE are symmetric and unaffected, but **R² is not**: `r2_score` normalizes by the variance of its first argument. With near-constant early predictions the denominator collapses and R² explodes negatively. This is visible in the committed artifacts: `experiment_results/20260217_100712_SimpleMLPRegressor/metrics.yaml:83-97` shows `test_r2_target_1: -156.12` next to `test_rmse_target_1: 1.20` (correct-order R² for that RMSE on z-scored targets would be ≈ −0.45), while the baseline R² in the *same file* (computed with correct order at `pipeline_simple.py:394`) is a sane ≈ 0.0. The **unscaled** validation metrics use the correct order (`pipeline_simple.py:359`), so scaled and unscaled val R² in one run are mutually inconsistent. Every scaled validation R² ever produced by both pipelines is invalid; any decision made by looking at them was made on noise.

### 3.2 Validation metrics are relabeled as "test" metrics — CRITICAL (reporting integrity)

`run_simple_tuning` returns `"test_metrics": {k.replace('val_', 'test_'): v ...}` with placeholder all-zero `test_preds`/`test_actuals` and `user_id: -1` (`pipeline_simple.py:131-151`). If `run_final_test` runs, these are overwritten; if it doesn't (crash, interrupt, or any future caller that only tunes), `metrics.yaml`, `predictions.csv`, and all plots present CV-tuning scores as test results. The committed February runs are exactly this case (`metrics.yaml:47-51`: `fold: 0, user_id: -1`, `test_* ≡ mean_val_*`). A results directory cannot be trusted without knowing which code path produced it, and nothing in the artifact records that.

### 3.3 Environment cannot run the code — CRITICAL (reproducibility)

`pyproject.toml:8-18` declares `torchinfo` but **not `torch`**; `poetry.lock` contains 38 packages, none of which is `torch` (verified). Every framework module imports `torch` at module load. `tabpfn` (a compared model) and `auto-sklearn` are also undeclared (install notes live in docstrings, `sklearn_models.py:249-273`). The actual experiments evidently ran in an undocumented Docker/venv; from this repository alone, `poetry install && main` fails at the first import. Additionally `os.environ["PYTORCH_ALLOC_CONF"]` (`main_simple.py:8`) is the wrong variable name for the CUDA caching allocator on the torch versions contemporary with this code (`PYTORCH_CUDA_ALLOC_CONF`), so the intended fragmentation mitigation is likely a silent no-op.

### 3.4 OOM fallback restarts training on a half-trained model — HIGH (latent)

`Trainer.train_model` wraps the whole multi-epoch GPU loop in `try/except RuntimeError` (`trainer.py:39-78`). If an OOM (or *any* `RuntimeError` — shape mismatch, device error) fires at epoch k, the handler re-runs the **full** `for _ in range(epochs)` loop via DataLoader on the *same* model and optimizer — continuing from the partially trained state with a fresh CosineAnnealing schedule. The affected fold silently trains up to 2× epochs with a restarted LR schedule; a genuine shape bug is masked into a slow wrong-training path instead of an error. The same over-broad `except RuntimeError` guards evaluation (`trainer.py:113-125`).

### 3.5 Unguarded NaN propagation path from features to loss — HIGH (data-dependent)

Chain: (1) the feature extractor defaults to `val = np.nan` and swallows exceptions (`timeseries_features.py:114-183`); rows missing a series contribute no key → NaN cell in the DataFrame; (2) missing `(user, path)` combinations become all-NaN rows via `reindex` (`dataset.py:147-155`); (3) sklearn's `StandardScaler` *ignores NaN in fit and preserves it in transform* — so NaN sails through `prepare_fold_data` into the tensors; (4) a single NaN in `x_path` makes the forward pass, the Huber loss, and every gradient NaN, silently zeroing an entire fold or trial (Optuna receives NaN); sklearn models instead hard-crash; LARS raises and is silently caught (§2.8). There is not a single `assert np.isfinite(...)`/`torch.isfinite(...)` in the pipeline. Whether this fires depends purely on dataset completeness — the code offers no guarantee either way.

### 3.6 Per-split tensor geometry: shapes are data-dependent, unchecked across splits — HIGH (latent)

`max_time`, the channel list, and `unique_paths` are all derived independently per split (`dataset.py:132,174-197`). Consequences: train/test `x_ts` differ in T (tolerated by CNN/LSTM but changing padding fractions systematically, §2.4); a channel-set difference between splits either crashes or — worse, if counts coincide — silently permutes channel semantics between train and test; a missing path for a test user changes `n_paths` and breaks every model that hard-codes `self.n_paths` from the *training* shape (`BASE_baseline_flat.py:31-33,63,104`; `mlp_path_specific.py:44-53`). `x_user`/`y` alignment relies purely on both CSVs being sorted and having identical user sets — there is no join or assertion (`dataset.py:127-143`).

### 3.7 `ElasticNetModel`'s defining hyperparameters are dead code — MEDIUM

The docstring says L1 must be added manually in the training loop (`framework/models/simple_models.py:157-165`), but `Trainer` only ever reads `lr`, `weight_decay`, `batch_size`, `eta_min` and applies Huber + AdamW (`trainer.py:14-29`). `alpha` and `l1_ratio` are sampled by Optuna (`simple_models.py:169-178`) and **never used** — the "ElasticNet" is an L2-regularized linear model, and part of its tuning budget optimizes no-op parameters. (`RandomForestLikeMLP` is likewise just a small MLP; naming only.) The saved `ElasticNetModel` results folder therefore mislabels the model family.

### 3.8 Loss values are not comparable across model families — MEDIUM

Torch models and the dummy baseline report Huber(δ=1) (`trainer.py:14`, `framework/models/dummy.py:11`); `SklearnTrainer.evaluate_model` reports plain MSE (`framework/core/trainer_sklearn.py:47-49`). On z-scored targets with |err|<1, Huber ≈ MSE/2, so every printed/saved "loss" comparison between a sklearn model and the baseline (or a torch model) is skewed by ~2×. RMSE metrics are unaffected.

### 3.9 Validation R² is never persisted; the paper-table generator reads keys that don't exist — MEDIUM

`saving.py` aggregates per-fold validation metrics filtered by `if "rmse" in metric_name` (`framework/reporting/saving.py:225-235`) — R²/MAE are dropped. `paper_comparison.py` then queries `val_r2_target_{i}` (`framework/reporting/paper_comparison.py:118-131`) → all val-R² columns in the paper CSV are NaN. (Given §3.1 those values were wrong anyway, but the plumbing is also broken.) Also `load_data` reorders columns using `target_names` leaked from the *last* loop iteration (`paper_comparison.py:135-141`) — incorrect ordering when experiments have heterogeneous target sets.

### 3.10 Early-stopping bookkeeping and shared-state mutations — LOW/MEDIUM

- No best-weight restore on early stop (§2.3, `trainer.py:99-106`).
- `run_final_test` does `final_results = tuning_results.copy()` (shallow) then mutates `final_results['fold_data'][0][...]` (`pipeline_simple.py:231-266`) — the tuning results dict is silently corrupted for any later reader.
- `total_model_parameters` is 0 for tuning-only runs (`pipeline_simple.py:112`; visible in `config.yaml:20`).
- `MLPSharedEncoderFLAT` defaults are outside their own search space (`pooling: "attention"`, `hidden_dim: 8` vs space `["flatten"]`, `[18,24,32]`, `mlp_path_shared_flat.py:148-177`) — 'default' mode evaluates an architecture the 'optimize' mode can never select.
- `HierarchicalTimeseriesLSTM._compute_lengths` counts nonzero entries instead of locating the last valid index (`LSTM.py:106-110`) — wrong even before scaling whenever a series contains internal zeros.
- `_impute_timeseries` places the imputed point at raw timestamp 0.0 because `start_time` is initialized but never assigned (`timeseries_imputer.py:171-189`); `impute_zero_disturbance` fabricates a 10 s duration when none is known (`timeseries_imputer.py:94-100`).
- Band-power indexing uses `np.argmax(freqs >= band[0])`, which returns 0 when no frequency qualifies (`timeseries_features.py:336-343`) — partially guarded, degenerate for 1-point/low-rate series.
- `dataset.py:216` falls back to column 0 (**raw timestamp**) as the value column for 2-column arrays; currently unreachable only because the loader filters those out (`timeseries_loader.py:176-179`).
- The active headline model `BASEBaselineFLAT` never touches `x_ts` at all (`BASE_baseline_flat.py:76-111`) — legitimate as an ablation, but note that the currently configured `main_simple.py:111` run therefore contains no time-series learning whatsoever while the repo README/paper foreground it.

---

## 4. Priority Recommendations & Mitigation Strategy

### CRITICAL — results-invalidating; fix before trusting or publishing any number from this pipeline

1. **Fix the swapped `calculate_metrics` arguments** at `pipeline_simple.py:354` and `pipeline.py:285` (order must be `(actuals, preds)`), then regenerate every saved metrics artifact; treat all historical scaled-val R² (and any tuning-only "test_r2_*") as void. Add a unit test that asserts asymmetric metrics against a hand-computed case.
2. **Make padding first-class**: persist true sequence lengths (or masks) in the dataset split, standardize `x_ts` using only valid timesteps (masked mean/std), and pass lengths explicitly into the models instead of re-inferring them from a sentinel the scaler destroys. Re-run all CNN/LSTM experiments; expect the published deep-vs-flat ranking to change.
3. **Decouple training decisions from the scoring fold**: either fix the epoch count (matching the final-training protocol) or early-stop on an inner sub-split; restore best weights on stop; make CV and final training protocols identical.
4. **Stop relabeling validation as test**: `run_simple_tuning` must return honestly named keys (`val_*`), and saved artifacts must record which pipeline stage produced them. Purge/rebuild the committed `experiment_results/` that contain placeholder test rows (`user_id: -1`).
5. **Restore honest generalization estimation**: for the headline claim, either (a) true nested CV (`optimize_every_fold`, with inner scaling fixed — `pipeline.py:83/93` must pass `scale_data=True` — and no `optimize_once`, no `only_first_fold`), or (b) keep Strategy B but report the tuning score explicitly as a model-selection objective and rely on an adequately sized, untouched test set for performance claims. Target pruning and model ranking must move inside whatever loop defines the reported estimate.
6. **Declare the real environment**: add `torch` (pinned, with CUDA variant), `tabpfn`, and the actual Python/Docker recipe to `pyproject.toml`/lock; CI-check that `main_simple.py` imports.
7. **Implement the imputation the paper describes, or amend the paper**: currently no fold-wise KNN(k=1) exists anywhere; the discrepancy is a reproducibility blocker for the publication itself. Same for the missing `user_force_total_mag` features (fix the `TS_FEATURES` key at `config.py:183` and make the extractor fail loudly on unknown keys) and for the aggregation-strategy search spaces that the paper says were optimized.

### HIGH — correctness/robustness hazards that can silently corrupt future runs

8. **NaN guards at every boundary**: assert finiteness after feature extraction, after scaling, and before loss; fail the fold/trial loudly, never silently (also remove the silent `except` in `lars.py:195` and narrow the `except RuntimeError` blocks in `trainer.py` to genuine OOM handling — and make the OOM fallback rebuild model/optimizer/scheduler instead of resuming a half-trained state).
9. **Cross-split consistency contracts**: persist and verify (channels, order, `max_time`, `n_paths`, path IDs, user alignment) as metadata written at split time and asserted at load time; align `x_user`/`x_path`/`y` by explicit joins on user/path keys, not sort order.
10. **Seed everything**: `torch`/`cuda`/`numpy`/DataLoader generators and `optuna.samplers.TPESampler(seed=…)`; log seeds into `config.yaml`; report mean±std over ≥5 seeds for the deep models, since at N=25 seed noise rivals model differences.
11. **Fix the time base of `x_ts`**: resample all channels onto one explicit per-path time grid (e.g., 1 Hz from motion start), with per-channel validity masks instead of row deletion/compaction; only then is a cross-channel Conv1d/LSTM defensible.
12. **Replace 1-point series imputation** with time-resolved imputation (interpolation within series where physiologically sensible) plus explicit missingness indicator features; never synthesize degenerate series that poison downstream statistics.

### MEDIUM — validity polish and hygiene

13. Unify the loss across trainers (or report RMSE only) so baselines and model families are comparable (`trainer_sklearn.py` vs `dummy.py`).
14. Remove or implement `ElasticNetModel`'s L1 (and rename `RandomForestLikeMLP`); align every model's defaults with its search space; drop no-op search dimensions.
15. Report headline metrics in unscaled units via per-fold inverse transform (already computed) rather than concatenated per-fold-scaled values; avoid pooled R² on LOO predictions or justify it explicitly.
16. Persist all validation metrics (not just `*rmse*`) in `metrics.yaml`; fix the `target_names` scoping and the NaN val-R² columns in `paper_comparison.py`; capture parameter counts for tuning-only runs.
17. Replace pseudo-replicated LARS/correlation selection with user-level aggregation (select on per-user path-averaged features) or a group-aware selector; make selection failures loud.
18. Quarantine legacy code (`archiv/`, `one_shots/`, `sklearn_test/`, `philipp_examples/`, the ~12 `hierarchical_attention_*` variants, `pipeline.py`+`main.py`) behind an explicit `legacy/` boundary with a README stating what produced which published number; delete dead imports (`pipeline.py:12`). Document that the legacy `bfill/ffill` (future-leaking), the ungrouped `MultiTaskLassoCV(cv=5)` (`dimred/lasso.py:19-23`), and the random 4-user test split (`sklearn_test/datasets.py:226-249`) are deprecated and why.
19. Add a minimal test suite (currently zero tests): metric-orientation, scaler/mask round-trip on a synthetic ragged batch, split-consistency assertions, one end-to-end smoke run on synthetic data. Most findings in this report would have been caught by ~5 such tests.

---

*Bottom line*: the architecture (fold-wise scaling, group-aware CV, selection-inside-fold, hold-out with inverse-transformed metrics) is fundamentally sound for an N=28 pilot, but the implementation currently (i) reports selection-biased and partly mis-computed validation numbers, (ii) defeats its own sequence masking before the deep models ever see the data, and (iii) cannot be re-run or exactly reproduced from what is committed. Fixing items 1–7 is a precondition for the next (clinical) data collection round; items 8–12 determine whether that round's results can be trusted without another forensic pass.
