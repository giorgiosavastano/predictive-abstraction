# Experiment 2 — Spatial granularity and information loss

All numbers are copied from `summary/tables.md` and `summary/*.csv`, regenerated from `raw/*.jsonl` by `uv run macro-analyze --experiment2 results/experiment2`. Mean ± SD across 10 seeds; paired 95% t-intervals.

## 1. Hypothesis

If coarse-graining helps prediction of the global target, there should be an intermediate number of spatial regions that minimizes error: too few regions discard useful spatial structure, too many re-introduce microscopic detail that a finite-resource learner cannot exploit. Alternatives: error decreases monotonically with resolution (detail always helps), or the three global summaries suffice (the regional features add nothing). A further question is whether the benefit comes from coarse-graining at all, or from the particular engineered global statistics.

## 2. Setup

- Same 10 seeds and splits as Experiment 1; within a seed every condition sees identical episodes.
- Resolutions 1, 2, 4, 8, 16, 32 (original `interior` edge rule) and 1–64 (`left` edge rule, which makes 64 bins well defined; see `docs/methods.md` §3 for what each resolution preserves and discards).
- Feature sets: `macro` (3 global + 3 per bin), `regional` (3 per bin only), `global` (3 features), `random` (projection with 3 + 3 × bins dimensions), `raw` (128).
- Main sweep: medium preset, budgets 250 / 750 / 2,000 / 5,000. Granularity × capacity grid: small, medium, large at 250 and 5,000. Local control: `macro` at every resolution and budget. Size proxies on the 5,000-episode pool. Permutation importance (validation split, 10 repeats) for the 8-bin macro model at 5,000.
- Command: `uv run macro-experiment2 --output results/experiment2` (27 min).

## 3. Metrics

Test / shifted / validation RMSE; number of features; marginal-entropy encoding bound (bits); effective rank (99% variance); mean absolute inter-feature correlation; feature-extraction, fit and inference seconds; which resolution wins on validation per seed.

## 4. Results

**Test RMSE vs resolution, medium preset (global task).**

| Bins | Features | macro, n=250 | macro, n=5,000 | regional-only, n=5,000 | random proj., n=5,000 |
| --: | --: | --: | --: | --: | --: |
| 1 | 6 / 3 / 6 | 0.451 ± 0.017 | 0.394 ± 0.008 | 0.684 ± 0.012 | 0.955 ± 0.033 |
| 2 | 9 / 6 / 9 | 0.456 ± 0.017 | 0.387 ± 0.008 | 0.484 ± 0.009 | 0.861 ± 0.044 |
| 4 | 15 / 12 / 15 | 0.455 ± 0.015 | 0.372 ± 0.008 | 0.407 ± 0.011 | 0.797 ± 0.028 |
| 8 | 27 / 24 / 27 | 0.456 ± 0.015 | **0.366 ± 0.010** | **0.397 ± 0.010** | 0.725 ± 0.031 |
| 16 | 51 / 48 / 51 | 0.459 ± 0.014 | 0.373 ± 0.009 | 0.441 ± 0.014 | 0.634 ± 0.019 |
| 32 | 99 / 96 / 99 | 0.468 ± 0.015 | 0.383 ± 0.007 | 0.492 ± 0.014 | 0.585 ± 0.020 |
| 64 (left rule) | 195 / 192 / 195 | 0.466 ± 0.013 | 0.387 ± 0.008 | 0.519 ± 0.010 | 0.538 ± 0.009 |
| global only | 3 | 0.457 ± 0.014 | 0.405 ± 0.009 | | |
| raw state | 128 | 0.820 ± 0.017 | 0.521 ± 0.012 | | |

The `interior` and `left` edge rules give indistinguishable results at every shared resolution (differences ≤ 0.003).

**Paired differences at 5,000 examples, medium (macro, vs 8 bins):** 1 bin +0.029 [0.025, 0.032]; 2 bins +0.021 [0.017, 0.025]; 4 bins +0.007 [0.003, 0.011]; 16 bins +0.007 [0.004, 0.010]; 32 bins +0.017 [0.014, 0.021]; global-only +0.040 [0.037, 0.042]; raw +0.156 [0.142, 0.170]. 8 bins is the validation-best resolution in 8/10 seeds (4 bins in the other 2). **At 250 examples no resolution differs from 8 bins** except 32 bins (+0.012 [0.006, 0.018]); 1 bin is nominally best (−0.005, p = 0.06) and is the validation-best choice in 4/10 seeds (7/10 at small capacity).

**Granularity × capacity (macro, test RMSE).** At 250 examples every cell lies in 0.450–0.472 regardless of capacity or resolution. At 5,000 examples: small preset is flat at 0.419–0.424; medium has its minimum at 8 bins (0.366); large has its minimum at 8–16 bins (0.330 / 0.330), with 4 bins at 0.348, 32 at 0.340, 64 at 0.350 and 1 bin at 0.411. The validation-best resolution moves finer as resources grow (1–2 bins at small/250; 8 at medium/5,000; 8 or 16 at large/5,000).

**Feature-set ablation.** Global-only (3 numbers) reaches 0.405 at 5,000, about 0.04 (11%) above the best medium macro model and 0.12 better than the raw state. Regional-only shows a pronounced interior optimum: 0.684 at 1 bin (it lacks the global interaction feature), 0.397 at 8 bins, back to 0.519 at 64 bins. Permutation importance of the 8-bin macro model: `global_disp_x_unreliability` 0.932 ± 0.030 RMSE increase when permuted, `global_displacement` 0.130 ± 0.013, `global_descents` 0.013, each regional feature ≤ 0.01 (regional displacement features sum to 0.67, regional unreliability to 0.48, regional descents to 0.02).

**Random projection control.** Error falls monotonically with projection dimension (0.955 at 6 dims → 0.538 at 195 dims) and only reaches the raw state's level at 195 dims; it never approaches any engineered representation of the same size.

**Representation size.** Marginal-entropy bound: global 19.5 bits, macro-8 85.7, macro-64 314, raw 273, random-27 216, random-195 1,559. Effective rank (99% variance): macro-8 23/27, raw 103/128. The 64-bin macro representation has a *larger* entropy bound than the raw state yet 26% lower error, so "fewer bits" is not what distinguishes the engineered features.

**Costs (5,000 examples, medium).** Extraction ≤ 3.7 ms for any representation; fit 0.19 s (global) to 0.27 s (macro-32) vs 0.32 s (raw) and 0.47 s (random-195); inference 4–9 ms. Cost grows mildly with feature count.

**Shifted split (5,000, medium):** macro-8 0.383 ± 0.016, macro-1 0.399, macro-32 0.402, global-only 0.417, raw 0.629, random-195 0.802.

**Local control task (macro, medium, 5,000):** 1 bin 6.695, 2: 4.999, 4: 4.862, 8: 4.757, 16: 4.573, 32: 4.280, 64: 4.150 (± 0.08–0.17). Finer is monotonically better; 32 or 64 bins is validation-best in 10/10 seeds; all remain far above the raw state (3.41).

## 5. Uncertainty

Across-seed SDs are 0.007–0.018 for the macro sweep; the resolution effects at 5,000 (0.007–0.029 vs 8 bins) are resolved by paired CIs, but at 250 they are not (CIs include zero for 1, 2, 4 and 16 bins). The regional-only and random-projection effects (0.1–0.5) are many SDs wide.

## 6. Interpretation

- **There is an interior optimum, but it is shallow and resource-dependent.** With the full macro feature set the U-shape spans only 0.366–0.394 at 5,000 examples and vanishes at 250. The optimum moves finer with more data and capacity, as a bias–variance account predicts. The clearest "less is more" signal is in the regional-only ablation, where 4–8 bins beat 64 bins by 0.12 RMSE despite using fewer features; no mutual-information comparison was measured.
- **Most of the benefit is not coarse-graining; it is the engineered global statistics.** Three global numbers reach an error about 11% above the best medium model; one of them (displacement × unreliability) carries most of the predictive weight. Even the finest engineered representation (64 bins, 195 features, a larger marginal-entropy bound than the raw state) beats the raw state by 26%. The regional features add a real but modest 10% at the largest budget and nothing at the smallest.
- **Random compression is harmful at every size.** The matched-dimension random projection is worse than the raw state at every dimension below 195 and worse than every engineered representation.
- **Task dependence is robust.** The local target prefers the finest available resolution in every seed and still prefers the raw state over any coarse-grained representation.

## 7. Limitations

- The resolution sweep uses one learner family and, in the capacity grid, only two budgets.
- The entropy bound is an upper bound on the discretized representation's entropy and is not a mutual-information estimate; no predictive-information estimates were attempted.
- The 64-bin point requires a modified edge rule; it is not the raw state (signs of displacement and element identities are discarded).
- Permutation importance is a model-specific measure on one resolution and budget.
- All conclusions concern supplied, hand-engineered abstractions.
