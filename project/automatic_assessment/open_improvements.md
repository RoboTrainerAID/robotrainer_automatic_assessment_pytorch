# Open Improvements (living document)

**Last updated:** 2026-07-08, sixth pass — `reporting/comparison.py` ported to schema v3 (+ two new diagrams: ratio summary with selection-gap overlay, per-fold distribution boxplots), the **augmentation-ratio experiment is analyzed (§1d: ratio 1 gives a small significant gain in the clean default comparison; per-ratio Optuna drowns in selection noise)**, and the project **README.md was rewritten** as the onboarding/architecture reference (old one kept as `README_old.md`).

*(Fifth pass: data augmentation implemented + R11; KNN-imputation note corrected.)*

*(Fourth pass: R1 + R2 + artifact schema v3; sweep analysis §3b — grouped masking pays off. Third pass: full-resolution channel groups, mean epoch budget.)*

---

## 0. Current state — what is implemented and verified

### Pipeline properties (all verified by `tests/test_pipeline_smoke.py` + real-data checks)

- **Correct metrics:** `calculate_metrics(y_true=…, y_pred=…)` everywhere; val R² is saved and matches a manual `r2_score`.
- **Honest reporting:** tuning produces only `val_*` results with real per-user LOGO-CV predictions; the optional final test ADDS `test_*` to a new dict. `predictions.csv` carries a `split` column; plots live in `plots/val/` and `plots/test/` with identical filenames.
- **Reproducibility:** global seeding + seeded Optuna sampler. Empirically confirmed: two identical runs (`20260706_132402` vs `20260706_133841`) are **bit-identical** in every metric.
- **Early stopping:** best-epoch weights are restored; per-fold `best_epoch` recorded; the final model trains for the **mean** best epoch across folds (changed from median on request).
- **Timeseries representation (NEW, third pass):** see §1 below. Old 1 Hz grid removed.
- **Scaling:** per-fold, per-channel masked standardization (statistics from valid bins only, padding re-zeroed); NaN/Inf hard-fails at load and after scaling.
- **Data hardening:** feature-config key validation, NaN checks in the validator, `process_dataset.py` fails loudly, restored Eq.-4 `user_force_total_mag_*` features (path features 78 → 88).
- **ElasticNetModel** actually applies its L1/L2 penalty; LARS failures are loud.

### Architecture refactorings implemented (2026-07-07, pass 4 — R1 + R2)

- **R1 — Typed result objects** (`framework/data/results.py`): `ExperimentConfig`, `MetricTable` (replaces the deleted `reporting/metrics.py` prefix-dict helper), `FoldResult`, `ValidationResult`, `TestResult`, `ExperimentInfo`, `ExperimentResult`. The pipeline fills them directly; saving/reporting only call their helper methods (`predictions_long()`, `metrics_nested()`, `per_target_frame()`, `fold_frame()`, `learning_curve_frame()`, `config_nested()`). `run_final_test` returns a NEW `ExperimentResult`; the tuning-only object is immutable by convention. The dummy baseline is now evaluated inside the same fold loop with the same scalers, which also yields the previously missing **validation baseline in original units**.
- **R2 — Central X schema** (`framework/data/schema.py`): single source of truth for the input layout (`X_PATH`/`X_USER`/`GROUPS_START` constants, `TSGroup` dataclass, `build_X`, `split_inputs`, `group_shapes`, `iter_group_tensors`, `input_shapes_info`). `masking.py` now contains only mask math; dataset, fold scaling, feature selection, all models, and the pipeline import the layout from schema. No index literals or parallel conventions remain.
- **Artifact schema v3** (user decision: format evolution allowed): `metrics.yaml` is nested (`val`/`test` × `scaled`/`unscaled`, per-target blocks keyed by **target name**, per-fold summaries, `schema_version: 3`); `predictions.csv` is **tidy long format** (one row per split/user/target with `y_true`/`y_pred` in scaled AND original units — replaces the wide file pair); `metrics_per_target.csv` replaces `metrics_unscaled.csv`; `config.yaml` nests `experiment` (the `ExperimentConfig`) + `info`. Visualization is v3-native **with a legacy fallback** that converts pre-v3 folders on the fly (verified against `20260706_132402`), so all existing result folders stay plottable/comparable. The orphaned attention plot (fed by artifacts nothing writes since the nested-CV era) was removed.
- Legacy models were moved to `archiv/` (user); `archiv/` no longer imports cleanly (it references the deleted `reporting/metrics.py`) — it is a frozen reference, not runnable code.

### Latest reference runs

| Run | Code state | val RMSE | test RMSE (n=3) |
|---|---|---|---|
| `20260706_100213` | pre-fix (old padding, no restore, 78 features) | 0.899 | 0.718 |
| `20260706_132402` / `133841` | post-fix, 1 Hz grid era, 88 features | 0.863 (bit-identical across both runs) | 0.919 |
| `20260706_17xxxx` sweep (9 models) | grouped full-resolution representation | **0.852** (CNN_BaselineNOEMBED, best) | 0.789 | 

Consistency check passed: `BASE_BaselineFLAT` (ignores ts tensors) reproduced **0.862831 bit-identically** in the sweep — pipeline integrity + seeding across code passes confirmed.

### 3b — Grouped-representation sweep analysis (2026-07-06 17:0x–17:18, 9 models, default params, seed 42) — did the grouped masking help? **Yes.**

| Model | uses ts tensors | val RMSE | val R² | test RMSE (n=3) | final ep. | CV duration |
|---|---|---|---|---|---|---|
| **CNN_BaselineNOEMBED** | ✔ | **0.852** | **0.378** | **0.789** | 11 | 1:36 |
| **LSTM_Baseline** | ✔ | **0.858** | 0.373 | 0.842 | 27 | 13:58 |
| BASE_BaselineFLAT | – | 0.863 | 0.367 | 0.919 | 11 | 0:50 |
| MLP_Baseline | – | 0.897 | 0.314 | 0.914 | 10 | 0:43 |
| CNN_Baseline | ✔ | 0.965 | 0.212 | 0.911 | 24 | 2:25 |
| SVR | – | 0.981 | 0.176 | 0.912 | 1 | 0:08 |
| BASE_Baseline | – | 0.989 | 0.174 | 0.922 | 24 | 1:27 |
| LinearRegression | – | 1.000 | 0.146 | 0.897 | 1 | 0:12 |
| ElasticNet | – | 1.018 | 0.115 | 0.885 | 1 | 0:09 |
| *(dummy baseline)* | | *1.089* | *≈0* | *0.979* | | |

**Readings:**
1. **Grouped full-resolution masking has a measurable benefit.** For the first time in the project's history, the two top validation models are the ones consuming the timeseries tensors — CNN_NOEMBED (0.852) and LSTM (0.858) beat the best hand-crafted-features-only model (BASE_FLAT, 0.863). In the paper era (1 Hz-equivalent representation, broken masking) the ts branch never beat "Base". The learned features now demonstrably add information beyond the 88 statistical features, and the same model also leads the 3-user test spot check (0.789) — directionally consistent.
2. **Consistency check passed** (see above): the grouping change did not disturb the path-feature pathway.
3. **CNN bottleneck artifact:** CNN_Baseline (0.965) vs CNN_NOEMBED (0.852) differ mainly in the tight `path_dim=12` bottleneck after concatenating ts embeddings with the 88 path features → the default bottleneck strangles the signal (see O1.2).
4. **Linear models invert between splits** (val ≈ dummy, test ≈ 0.89): 20×88 flattened features overfit ungrouped linear models per fold, while the "average" test users flatter them — the familiar normative-test-set effect; do not read the linear test scores as skill.
5. Caveat: single seed, default hyperparameters — the 0.852 vs 0.863 lead needs the O1.1 seed study before declaring a winner.

---

## 1. The new timeseries representation (answers to your questions)

**Why the 1 Hz grid was wrong for your goal (agreed):** averaging everything to 1 Hz reproduces information the statistical features already capture and destroys exactly the fast dynamics (force-area hits, tremor band 3–10 Hz) that learned features are supposed to discover. It optimized alignment — which you don't need — at the cost of resolution — which you do.

**What is implemented now** (`config.TS_MODEL_GROUPS`, `framework/data/dataset.py`):

- **Three channel groups, each with its own tensor, rate, and length:** mechanical (21 ch) @ **50 Hz** (200 Hz force bin-averaged 4:1 — Nyquist 25 Hz keeps the voluntary AND tremor bands intact), physiological (3 ch) @ **2 Hz** (native HRV/PPI resolution, HR simply has masked gaps), gait events (8 ch) @ **2 Hz** (~1 event/stride). Groups are NOT aligned to each other — per-group learned features, as you wanted.
- **Variable length is handled without downsampling:** every sequence keeps its true length inside the group tensor; a boolean mask marks valid bins. **LSTMs use `pack_padded_sequence` with mask-derived lengths — the recurrence literally skips the padded tail** (that IS the "smart masking that skips unnecessary length"). CNNs convolve the padded region (cheap, parallel) but pool with the mask, so padding never influences the output. This is the standard, correct way to give CNNs/LSTMs variable-length input; no per-sample dynamic architectures needed.
- **Where the resampling lives:** nowhere near the statistical features (they keep using the raw full-resolution npy in `process_dataset.py`), and not in the training loop. The grouped tensors are **precomputed into each split folder** as `ts_tensors_cache.npz` — written when you run `framework/data/dataset.py` (split creation) and self-healing on first `AssessmentDataset` load if absent or stale (data/config fingerprint). Regenerating the splits regenerates the cache automatically.
- **Real-data footprint:** mechanical `(25,20,21,4040)` — full 50 Hz over the longest path — 212 MB; physiological `(25,20,3,161)`; gait `(25,20,8,150)`; total 216 MB, cache build 2.3 s, cached load 0.4 s.
- **Models reworked:** `CNNBaseline`, `CNNBaselineNOEMBED(+FLAT)`, `LSTMBaseline` have one encoder per group (shared hyperparameters across groups, embeddings concatenated); `HierarchicalTimeseriesLSTM` runs its shared single-channel LSTM over every channel of every group. Verified with 1-epoch real-data runs: CNN 0.5 s/epoch, LSTM 1.3 s, per-channel LSTM 7.7 s (see O1 for its cost).

**Trade-offs to be aware of** (deliberate, revisit if needed): bin-averaging 200→50 Hz is a box filter (mild spectral leakage above ~15 Hz — fine for band-power-style content, not for precise spectra); `user_work_cum` is a cumulative signal whose per-channel z-scoring is dominated by path length; the mechanical tensor length is set by the longest path (78.6 s), so short paths are ~85% masked padding — harmless for correctness, slightly wasteful for CNN compute.

## 1b. Data augmentation (implemented 2026-07-07)

**Design** (all four decision points confirmed by you):
- **Semantics:** a clone is a *noisy re-measurement of the same person*. Targets, demographics, and task difficulty stay **bit-identical**; only the raw sensor channels receive noise.
- **Noise:** per (path, channel), `sigma = noise_factor × robust (IQR-based) std` of that trace — outlier spikes don't inflate it, and constant traces (zero-disturbance series) stay exactly constant. Positive-only channels (physiological + gait) get **multiplicative, truncated** noise so heart rates and stride times can't go negative. Derived series (magnitudes, power, work) are **re-derived** from the noisy raw components and all 85 statistical features are **re-extracted** from the noisy traces — the clone is physically self-consistent. Method is config-selectable (`jitter` now; `time_warp`/`magnitude_warp` can be added under the same interface).
- **Leak policy:** clones are generated ONCE at split creation, **only for training users** — a test-user clone can never exist. Within CV, `AugmentedLOGO` already excludes the validation user's clones from the training fold. Clone id = `original*100 + index`, each with its own deterministic RNG stream (`config.AUGMENTATION['seed']`).
- **Ratio mechanics:** the train split stores clones up to `config.AUGMENTATION['max_ratio']` (currently 5); experiments pick `AssessmentDataset.view(targets=…, augmentation_ratio=r)` — the ratio sweep `[0..5]` needs **no data regeneration**. Note `batch_size=6` keeps dividing the fold-train size for every ratio (24·(1+r)).
- **Config:** `dataset/config.py → AUGMENTATION = {max_ratio, method, noise_factor, seed}`.
- **R11 alongside:** `perform_target_selection` (mutating, restore-from-orig pattern) and the augmentation placeholder are gone; `dataset.view(...)` returns an immutable `DatasetView`.

**How to run the ratio experiment:** set `augmentation_range = [0, 1, 2, 3, 4, 5]` in `main_simple.py` (everything else unchanged). Each ratio gets its own result folders; `comparison.plot_augmentation_ratio_comparison` needs the O8 port first or a quick manual aggregation of `metrics.yaml`.

**What to expect / caveats (rank-2 method, see the option ranking in the 2026-07-07 discussion):** clones add no new label information — they encode a *sensor-noise invariance prior*. Benefits are expected mainly for the NN branch; for LARS/linear models clones pseudo-replicate rows (interacts with O6 — the selection sees `(1+r)×` copies of nearly identical evidence). If ratio>0 hurts validation, that is a legitimate finding (the invariance may already be satisfied), not a bug. The scientifically stronger long-term option remains on-the-fly input augmentation in the training loop (rank 1), which would also compose with warping methods.

## 1d. Augmentation-ratio experiment results (2026-07-07 runs, analyzed 2026-07-08)

BASE_BaselineFLAT, 8 targets, seed 42; ratios 0–5 in two modes. Full tables/plots: `experiment_results/augmentation_{default,optimize}/comparison_results/`.

**Default mode (identical hyperparameters across ratios — the clean comparison):**

| ratio | val RMSE | test RMSE (n=3) | users improved vs r0 | Wilcoxon p (fold-paired) | final epochs |
|---|---|---|---|---|---|
| 0 | 0.8628 | 0.9185 | – | – | 11 |
| **1** | **0.8436** | 0.8952 | **17/25** | **0.030** | 6 |
| 2 | 0.8475 | 0.8006 | 15/25 | 0.23 | 5 |
| 3 | 0.8530 | 0.7844 | 15/25 | 0.38 | 4 |
| 4 | 0.8845 | 0.8266 | 13/25 | 0.69 | 4 |
| 5 | 0.8698 | 0.7638 | 14/25 | 0.67 | 3 |

**Readings:** (1) **Ratio 1 is a real but small win** — the only statistically supported improvement (p=0.03, 17/25 users better); the noise-invariance prior helps once, additional near-identical clones add nothing and ratios 4–5 slightly hurt validation. (2) `final_epochs` drops 11→3 with ratio — each epoch sees (1+r)× batches, so convergence shifts into fewer epochs; the *effective* optimization budget is roughly constant. (3) Test RMSE trends down with ratio (0.92→0.76) — directionally interesting (regularization flattering the "average" test users) but n=3.

**Optimize mode (30 fresh Optuna trials per ratio — confounded by selection noise):** val RMSE bounces between 0.800 (r5) and 0.869 (r3) with no consistent ratio effect; all fold-paired comparisons vs r0 are non-significant or negative. The **selection gap is directly visible**: in 5 of 6 runs `best_trial_val_loss` < re-evaluated `val_loss` (e.g. r1: 0.267 selected vs 0.302 re-evaluated ≈ 13% optimistic), at r5 the re-eval happened to beat the best trial — noise in both directions, exactly the winner's-curse mechanism from §2/O4.

**Recommendation:** use **ratio 1 as the default** for model development runs (or 0 for maximum simplicity); do NOT tune per ratio; revisit augmentation with warping methods (rank-3 option) only if the NN branch stalls.

## 1c. Imputation correction (2026-07-07)

Confirmed by you: **no KNN imputation exists anywhere** (also not in the rosbag→numpy step) — the paper's sentence was wrong, deleted on purpose during development. Assessment: the no-value-imputation design is *correct* for this project — raw series + explicit masks for the learned branch, statistical features from real samples only; nothing in the inputs is fabricated. The only imputation that exists is the whole-missing-series 1-point stub (tracked in O2). Consequence recorded: paper wording is wrong on this point (paper stays as decided); future texts should describe masking, not imputation.

---

## 2. Your early-stopping question, answered (why not early-stop on the test run)

Early stopping needs a signal to *decide* when to stop; whatever data provides that signal is **choosing the model**. If the final model is evaluated every epoch on the test set and you keep/report the best-so-far test score, the reported number is `min over ~30 epoch-checkpoints of (test loss on 3 users)`. That minimum is an order statistic of noisy evaluations — it is biased low for exactly the same reason the best Optuna trial is biased low (winner's curse). The epoch has become a hyperparameter *fitted to the test set*, so the test set is no longer unseen — "leak" here doesn't mean gradients flowed from test data, it means a **decision** flowed from test data into the reported model. With n=3 users the effect is large, and the future-cohort test would inherit the same problem if the habit sticks.

What is implemented instead (your fallback): the final model trains for the **mean of the per-fold best epochs** — an epoch budget derived purely from training-side information. Remaining alternatives if the val↔test disagreement persists: a fixed CV-agnostic epoch count for all models (cleanest comparability), or higher patience to de-noise the per-fold stopping signal (see O3).

Related: `best_trial_val_loss: null` in your configs is expected — that field only exists in `optimize` mode (it is the Optuna best-trial score, to compare against the re-evaluated `val_loss`). In `default` mode there is no search, hence no selection gap to measure. Your seed/fold variance tests cover the rest of the former P1 concern; the winner's-curse caveat only becomes active again when you run `optimize` campaigns (then: equal `n_trials` per model, watch the `best_trial_val_loss − val_loss` gap, prefer multi-seed re-evaluation for reported numbers).

---

## 3. Open problems, ranked by importance

### O1 — Follow-ups from the grouped-representation sweep (see §3b) — HIGH, scientific
The default sweep is done (§3b): **two TS models now beat every path-features-only model on validation** — the first direct evidence that learned full-resolution features add information beyond the 88 hand-crafted features. Next steps:
1. **Seed-robustness of the new ranking**: run the top 3–4 (CNN_NOEMBED, LSTM_Baseline, BASE_FLAT, MLP_Baseline) with 3–5 seeds; paired per-fold comparison (O5). The 0.85-vs-0.86 gap is small.
2. **CNN bottleneck finding**: `CNN_Baseline` (path bottleneck `path_dim=12` squeezing 24 ts dims + 88 path features) scores 0.965 vs its no-bottleneck sibling's 0.852 — the default bottleneck is too tight for the grouped features. Widen `path_dim` in its search space or drop the model in favor of NOEMBED.
3. `optimize` campaigns for the leaders (equal `n_trials`, O4 rules apply).
4. Cost hotspot unchanged: `HierarchicalTimeseriesLSTM` (not in the sweep) loops 32 channels in Python (~7.7 s/epoch → 30-epoch CV ≈ 1.5 h); batch the channel loop before including it in sweeps. `LSTM_Baseline` at 50 epochs costs ~14 min per CV — fine.

### O2 — Physiological/gait sparsity and the 1-point imputations — HIGH, data/model
On the 2 Hz grids the physio/gait groups have paths with as little as **1 valid step** (the whole-series imputations that replace a missing series with a single averaged point). The encoders see an almost fully masked sequence whose pooled embedding is dominated by one value. Options: (a) missingness-indicator features per series (`<series>_was_imputed`), (b) impute as a constant series over the path duration instead of one point, (c) let models learn a per-group "no data" embedding when coverage < threshold. Also decide whether `user_work_cum` (cumulative) belongs in the mechanical group or should be dropped/differenced (its z-scored values mostly encode elapsed time).

### O3 — Epoch-budget policy — MEDIUM, methodological
Mean-of-best-epochs is implemented, but the per-fold best epochs still spread 1–24 because each fold's stopping signal is a single user (patience 2). If val↔test keeps disagreeing: fixed epoch count for all models (best comparability), or patience 3–4, or epoch selection on aggregated fold curves (average val-loss curve across folds, pick its argmin) — the latter is the most stable and still test-free.

### O4 — Hyperparameter-selection bias when `optimize` campaigns resume — MEDIUM, methodological
Dormant while running `default` mode. When tuning: equal `n_trials` per model; report the re-evaluated `val_loss` (already automatic) rather than `best_trial_val_loss`; multi-seed the winner; keep nested CV in reserve for final claims. (Detailed mechanism: §2 of this file, winner's curse.)

### O5 — Statistical reporting standard for model comparisons — MEDIUM
Per-fold val losses and best epochs are now saved (`fold_metrics`); use them: paired per-fold comparisons (e.g. Wilcoxon) and mean±std over seeds instead of single-run rankings. Your ongoing variance tests slot directly in here.

### O6 — Feature-selection pseudo-replication — MEDIUM (grows with augmentation)
LARS/correlation selection still flattens `(N,P,F) → (N·P,F)` and repeats each user's target 20× — fold-internal (no leakage) but statistically overconfident. **With augmentation ratio r this multiplies further**: the selector sees `(1+r)` near-identical copies of every user's evidence (at r=5 a fold-train of 24 users becomes 2880 pseudo-samples of 24 people). Options: select on per-user aggregated features, restrict selection to the ORIGINAL users of the fold (clones excluded from selection but kept for training), or a group-aware criterion.

### O7 — Data quality follow-ups — MEDIUM
(a) Minimum path duration **0.94 s** — one essentially empty recording; identify (analyse_ts.py) and fix/exclude upstream. (b) 44 whole-series imputations (loudly logged) — interacts with O2. (c) Mechanical tensors are ~85% padding for short paths because T is set by the longest path — correct but wasteful; if CNN compute ever matters, bucket paths by length. *(Closed per your note: the upstream KNN imputation is per-user and leaks nothing; paper wording stays.)*

### O8 — Reporting scripts vs. current schema — LOW-MEDIUM (partially resolved 2026-07-08)
`comparison.py` is **ported to v3** (with a legacy fallback, tidy tables, new ratio-summary + fold-distribution diagrams) and drove the §1d analysis. Still legacy-keyed: `paper_comparison.py`, `paper_comparison_plot*.py`, `paper_comparison_table.py`, `hyperparameter_analysis.py` — port or retire them before the next paper-style aggregation (R5 remainder).

### O9 — Legacy model files will crash if revived — LOW
`hierarchical_attention_*`, `hierarchical_cnn.py`, `cnn1d.py`, `mlp.py` (timeseries folder) and everything in `archiv/` still use pre-mask 3-tuple inputs and value-sentinel lengths. Delete or migrate before reuse; until then they are dead weight (see also refactoring item R8).

### O10 — Docs — RESOLVED (2026-07-08)
`README.md` rewritten as the full onboarding/architecture reference (quick start, data pipeline, ML pipeline, model list, project context, challenges, contributor conventions); the historical version is preserved as `README_old.md`. Keep README + this file in sync going forward.

### O11 — Test depth — LOW (foundation exists)
Smoke test covers the grouped pipeline synthetically + 1-epoch real-data model checks were run manually. Worth adding: a marked-slow 2-user real-data mini-CV, split-integrity tests for `prepare_and_split_data`, and cache-invalidation tests.

---

## 4. Code structure improvements (proposed only — awaiting your go/no-go per item)

Requested architecture review of the whole framework; ranked by expected payoff. **R1 and R2 are IMPLEMENTED (2026-07-07, see §0)** — the rest awaits your go/no-go per item.

- ~~**R1 — Typed result/config objects instead of nested dicts.**~~ **DONE** — `framework/data/results.py` (ExperimentConfig, MetricTable, FoldResult, ValidationResult, TestResult, ExperimentInfo, ExperimentResult) used end-to-end; `reporting/metrics.py` deleted.
- ~~**R2 — Central data-schema module.**~~ **DONE** — `framework/data/schema.py` is the single layout source; `masking.py` reduced to mask math; no index literals outside schema.
- ~~**R11 — Stateless dataset views.**~~ **DONE** (2026-07-07) — `AssessmentDataset.view(targets, augmentation_ratio)` returns an immutable `DatasetView`; the mutating `perform_target_selection` and the augmentation placeholder are removed.
- **R3 — Model registry + experiment spec.** Replace the comment-block model lists and hard-coded `models_to_test`/`targets_to_test` in `main_simple.py` with a name→class registry and a small experiment spec (YAML or CLI: `--models cnn base_flat --mode optimize --seeds 3`). Makes sweeps (O1, O5) one command and self-documenting.
- **R4 — Unify the two trainers behind one interface.** `Trainer` vs `SklearnTrainer` duck-type today; an explicit ABC (fit/evaluate/best_epoch/cleanup) plus an extracted `EarlyStopping` policy object would let the epoch-budget policy (O3) be swapped without touching trainer internals.
- **R5 — Reporting consolidation.** One documented metrics-schema + one aggregation tool replacing the four overlapping `paper_comparison*` / `comparison` / `hyperparameter_analysis` scripts (O8 becomes moot). Result folders get a `schema_version` field.
- **R6 — Feature-extraction registry.** The if/elif chain in `timeseries_features._compute_ts_features` becomes a `{name: function}` registry — per-feature unit tests, and unknown keys become impossible instead of validated-at-runtime.
- **R7 — Path/environment hygiene.** Hard-coded `/data` and `/workspace/...` paths → a single paths module reading env vars with defaults; makes the repo runnable outside this exact Docker layout and testable with fixture dirs.
- **R8 — Dead-code removal.** Delete or archive the ~10 unmaintained timeseries model variants (O9), `dimred/lasso.py` (unused `select_features`), and prune `archiv/` — every layout change currently drags a false sense of 60 model files.
- **R9 — Logging.** Replace the print/tqdm.write mix with the `logging` module (levels, per-run logfile in the results folder) — the imputation/validation reports would then be preserved artifacts instead of scrollback.
- **R10 — Pytest + CI.** Promote the smoke test to pytest, add ruff (and optionally mypy on `framework/`), run on every change. All of this week's silent-failure classes (swapped args, index drift, import breaks) are CI-catchable.
- **R11 — Stateless dataset views.** `perform_target_selection` mutates the dataset object between experiments (restore-from-orig pattern); returning an immutable view would remove the hidden state and the augmentation-placeholder coupling.

---

## 5. Log of resolved items (for traceability)

- 2026-07-06 pass 1: metric argument order (3.1), aligned+masked tensors (2.4/2.5 v1), best-weight restore (2.3), OOM restart (3.4), val-first results schema (3.2/3.10), seeding (2.9), ElasticNet penalty (3.7), LARS loudness (2.8), Eq.-4 feature key + validators (2.11/3.5), pyproject deps (3.3), smoke test.
- 2026-07-06 pass 2: splits regenerated (88 path features), first post-fix runs, plots restructured to `plots/val|test/`, `poetry.lock` regenerated (user).
- 2026-07-06 pass 3: 1 Hz grid replaced by grouped full-resolution tensors + per-group encoders (user decision, §1); mean epoch budget (user decision, §2); former P1/P2/P6 closed per user notes; accidental IDE move of `framework/data/` → `framework/dimred/data/` detected and reverted (stale copies backed up to the session scratchpad).
- 2026-07-07 pass 4: R1 + R2 implemented; artifact schema v3 (nested metrics, tidy predictions, legacy fallback in visualization); `reporting/metrics.py` deleted (archiv/ now a frozen, non-importable reference); orphaned attention plot removed; 9-model sweep analyzed (§3b) — grouped masking benefit confirmed.
- Historical-results validity: all pre-fix `val_rmse` values remain comparable *within the pre-fix era*; scaled val R² before pass 1 was never valid; runs before pass 3 used a different ts representation and are not comparable to current runs. Artifacts before pass 4 use the flat pre-v3 schema (readable via the visualization fallback).
