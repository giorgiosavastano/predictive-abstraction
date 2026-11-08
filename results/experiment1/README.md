# Experiment 1 — Robustness across seeds, data budgets and model capacities

All numbers below are copied from `summary/tables.md` and `summary/*.csv`, which are regenerated from `raw/*.jsonl` by `uv run macro-analyze --experiment1 results/experiment1`. Values are mean ± SD across 10 seeds unless stated; intervals are 95% t-intervals of paired across-seed differences.

## 1. Hypothesis

The baseline (one seed) found that a 27-feature hand-engineered macroscopic representation predicts global remaining displacement with lower error than the 128-feature microscopic state. The hypothesis under test is that this advantage is a real finite-resource effect: it should (a) replicate across independent seeds, (b) hold for a dimension-matched random projection control and the scalar baseline, and (c) persist, or at least shrink gracefully, as training data and model capacity grow. If the advantage disappears when the raw-state model is given more capacity and the freedom to choose its capacity on validation data, the baseline interpretation is falsified.

## 2. Setup

- 10 seeds (1000, 2000, …, 10000). Per seed: 5,000-episode training pool, 1,000 validation, 2,000 in-distribution test and 2,000 shifted-fault test episodes. Budgets 250 / 750 / 2,000 / 5,000 are nested prefixes of the pool, identical for all representations and capacities.
- Representations: Full state (128), Macroscopic (27, 8 bins), Random projection (27, fitted on the budget's training subset), Initial disorder (1), Full + macro (155).
- Learner: `HistGradientBoostingRegressor`, presets small (50 trees × 4 leaves), medium (100 × 9, the baseline), large (300 × 31), xlarge (1,000 × 63; 5,000 budget only, global task only). Learning rate 0.075, min leaf 12, L2 2.0, no early stopping.
- Tasks: global remaining displacement after 12 sweeps (primary); local centre value after one sweep (control; small/medium/large).
- Capacity selection: per seed, budget and representation, the preset with the lowest *validation* RMSE; test data never used for selection.
- Irreducible error: 32 re-simulations per test episode.
- Command: `uv run macro-experiment1 --output results/experiment1` (35 min on a 10-core Apple Silicon laptop).

## 3. Metrics

Test RMSE (positions per element) on the in-distribution split; shifted-test RMSE; R²; validation RMSE; wall-clock feature-extraction, fit and inference seconds; realised model size (trees, leaves, nodes).

## 4. Results

**Global task, test RMSE (mean ± SD over 10 seeds).**

| Capacity | Budget | Full state | Macroscopic | Random projection | Initial disorder | Full + macro |
| :-- | --: | --: | --: | --: | --: | --: |
| medium | 250 | 0.820 ± 0.017 | 0.456 ± 0.015 | 0.900 ± 0.035 | 0.779 ± 0.019 | 0.466 ± 0.018 |
| medium | 5,000 | 0.521 ± 0.012 | 0.366 ± 0.010 | 0.725 ± 0.031 | 0.734 ± 0.013 | 0.375 ± 0.008 |
| large | 250 | 0.807 ± 0.020 | 0.464 ± 0.017 | 0.898 ± 0.038 | 0.812 ± 0.023 | 0.469 ± 0.020 |
| large | 5,000 | 0.401 ± 0.010 | 0.332 ± 0.007 | 0.640 ± 0.020 | 0.744 ± 0.013 | 0.336 ± 0.008 |
| xlarge | 5,000 | 0.420 ± 0.010 | 0.341 ± 0.010 | 0.644 ± 0.021 | 0.745 ± 0.013 | 0.349 ± 0.008 |
| validation-selected | 250 | 0.807 ± 0.020 | 0.455 ± 0.014 | 0.897 ± 0.036 | 0.760 ± 0.016 | 0.461 ± 0.016 |
| validation-selected | 5,000 | 0.401 ± 0.010 | 0.332 ± 0.007 | 0.641 ± 0.020 | 0.731 ± 0.013 | 0.336 ± 0.008 |

Estimated irreducible RMSE of the global target: **0.192 ± 0.003** (target SD 1.176).

**Paired difference Macroscopic − Full state (test RMSE), 10/10 seeds favour macro in every cell.**

| Setting | Mean diff | 95% CI | Relative |
| :-- | --: | :-- | --: |
| small, 250 | −0.430 | [−0.441, −0.419] | −48.5% |
| medium, 5,000 (baseline setting) | −0.156 | [−0.170, −0.142] | −29.9% |
| large, 5,000 | −0.070 | [−0.079, −0.060] | −17.4% |
| xlarge, 5,000 | −0.079 | [−0.091, −0.068] | −18.8% |
| validation-selected, 250 | −0.353 | [−0.369, −0.337] | −43.7% |
| validation-selected, 5,000 | −0.070 | [−0.079, −0.060] | −17.4% |

Other paired comparisons (test RMSE): Macroscopic beats the random projection by 0.30–0.48 (46–52% relative) in every cell and every seed; beats the scalar baseline by 0.30–0.41; and is slightly *better* than Full + macro in every cell (by 0.001–0.012, macro lower in 8–10 of 10 seeds). On the shifted split the macro advantage is larger: at large capacity and 5,000 examples the paired difference is −0.145 [−0.158, −0.132] (−30%).

**Local control task (centre value after one sweep), test RMSE at 5,000 examples, medium capacity:** Full state 3.413 ± 0.118, Full + macro 3.424 ± 0.112, Macroscopic 4.757 ± 0.077, Random projection 5.187 ± 0.146, Initial disorder 6.684 ± 0.186; irreducible RMSE 3.013 ± 0.102. Paired Macroscopic − Full state = +1.344 [1.291, 1.397] (+39%); the full state wins in 10/10 seeds at every budget and capacity, and its margin *grows* with capacity (+24% at small/250 to +43% at large/5,000).

**Costs at 5,000 examples (means over seeds).** Feature extraction is negligible for every representation (0.4 ms raw, 2.6 ms macro, 3.1 ms random projection for 5,000 episodes). Fit time is set by capacity and, secondarily, by feature count: medium 0.44 s (full) vs 0.30 s (macro); large 4.03 s vs 3.20 s; xlarge 16.6 s vs 17.4 s. Inference on 2,000 episodes is 6–30 ms. Realised model sizes are identical across representations at a given preset (900 leaves at medium, 9,300 at large, 63,000 at xlarge), so comparisons at a preset are matched in parameter count as well as in hyperparameters. In a measured cost/error comparison, rather than a fixed-time training experiment, macro at medium (0.31 s, RMSE 0.366) beats full state at large (4.06 s, RMSE 0.401), 13× cheaper.

## 5. Uncertainty

Across-seed SDs are 0.007–0.038 for every global-task cell, far smaller than the macro–full gaps at low budgets (0.35–0.43) and still 5–7× smaller than the gap at the largest budget and capacity (0.070, SD of the paired difference 0.013). Paired t-tests give p < 1e−6 and Wilcoxon signed-rank p = 0.002 (the minimum for n = 10) in every cell. The irreducible-error estimate has across-seed SD 0.003.

## 6. Interpretation

- **The baseline replicates.** The baseline seed's numbers (full 0.502, macro 0.367 at medium/5,000) sit inside the 10-seed ranges (full 0.504–0.540, macro 0.350–0.382).
- **The advantage is real but strongly resource-dependent.** It is ~45–49% at 250 examples regardless of capacity, and shrinks to 17% at 5,000 examples once the raw-state model is allowed the large preset. The raw-state learning curve is still falling steeply at 5,000 examples (0.494 → 0.401 from 2,000 to 5,000 at large) while the macro curve is flattening (0.374 → 0.332); both are still well above the 0.192 floor. The declining gap motivates testing larger budgets. The information ordering concerns Bayes-optimal risk; it does not guarantee convergence of this particular learner.
- **Bigger is not automatically better.** The xlarge preset is worse than large for every representation (e.g. full 0.420 vs 0.401), a sign of over-fitting with a fixed learning rate and no early stopping. Validation selection picks large for both representations at 2,000 and 5,000, so the "fair" comparison at the top budget is the large-vs-large one.
- **Dimensionality reduction alone does not reproduce the advantage.** The 27-dimensional random projection has higher mean error than the summary and full state at every tested budget and capacity. It is worse than the disorder scalar at the smallest budget, but not in every setting.
- **Adding the raw state to the macro features does not help**, and marginally hurts, at every budget and capacity. Supplying the raw coordinates alongside the summary does not improve average performance for this learner at these budgets; this does not establish that the raw state has no additional predictive information.
- **The preferred representation depends on the target.** For the local task the ranking inverts in every seed, and the raw state's margin grows with capacity.
- **Robustness to shift** favours the macro features (30% lower shifted-test RMSE at large/5,000).

Overall: the hypothesis that a task-relevant engineered representation improves finite-resource prediction survives ten seeds, three to four capacities and validation-based capacity selection. The stronger reading one might be tempted to make, that the compressed representation is *better* in an absolute sense, is contradicted by the shrinking gap and by the distance of both curves from the noise floor.

## 7. Limitations

- One learner family (histogram gradient boosting) and one capacity axis. A locality-aware neural predictor on the raw state was not tested (future work, pre-registered in `docs/next-experiments.md`).
- The largest preset (xlarge) was run only at 5,000 examples and only for the global task, for cost reasons, and over-fits under the fixed learning rate; a learning-rate/early-stopping sweep might let a very large raw-state model close more of the gap.
- Wall-clock timings are single-machine measurements under default OpenMP threading; they rank configurations but are not portable benchmarks.
- The macro features embed knowledge of the sorted target; this experiment tests supplied abstractions, not discovered ones.
- The shifted split alters fault width and severity only.
