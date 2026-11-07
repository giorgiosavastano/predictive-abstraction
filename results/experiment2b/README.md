# Experiment 2b — Emergent predictive structure (approximate closure test)

All numbers are copied from `summary/tables.md` and `summary/*.csv`, regenerated from `raw/*.jsonl` by `uv run macro-analyze --experiment2b results/experiment2b`. Mean ± SD across 10 seeds; paired 95% t-intervals. No learner is involved anywhere in this experiment.

## 1. Hypothesis

Experiments 1–2 showed that a few collective variables are *useful to a learner*. The emergence hypothesis makes a stronger claim about the *system*: the sorting dynamics approximately close over those variables, so that microscopic configurations with the same macrostate have the same macroscopic future, P(Z_{t+τ} | X_t) ≈ P(Z_{t+τ} | Z_t). The alternative is that the variables are convenient coordinates for a tree ensemble while the future still depends substantially on microscopic detail. The two predictions differ in a learner-free quantity: the variance of the true conditional mean of the future that remains once the macrostate is fixed.

## 2. Setup

- Per seed, 10,000 configurations drawn from the episode distribution and advanced by 2 sweeps (so a one-sweep history exists and configurations are mid-recovery); 16 independent futures each, recording the three global summaries after every sweep up to τ = 12. Streams: `seed + 30` (configurations), `+31` (futures), `+32` (projections).
- Candidate abstractions: initial disorder (1), global (3), macro at 1/2/4/8/16/32 bins (6–99), regional-8 (24), random projections of 3/6/27/99 dimensions, raw state (128).
- Estimator: for each configuration, the k = 10 nearest *distinct* configurations in standardized φ-space; excess = mean over pairs of (μ_i − μ_j)²/2 minus realisation noise; closure = 1 − excess / V_total, where V_total is the (noise-corrected) variance of conditional means across configurations. Future measured as the target (mean displacement) and as the three global summaries (`global3`).
- Refinements: the 50 nearest in φ re-ranked by raw-state distance (`+raw`), by the previous sweep's global summaries (`+history`), by the present global summaries (`+global`, control for history), or at random (`+random`, control for any re-ranking).
- Convergence check: the plain estimator at 2,500, 5,000 and 10,000 configurations.
- Command: `uv run macro-experiment2b --output results/experiment2b` (10 min).

## 3. Metrics

Closure index and excess variance per abstraction, horizon, measure and sample size; relative neighbour distance (mean neighbour distance / mean pairwise distance in φ-space); V_total and V_irr by horizon; paired across-seed differences.

## 4. Results

**Closure at τ = 12 (target), full sample, mean over 10 seeds; SDs ≤ 0.004 except where shown.**

| Abstraction | Features | Rel. neighbour distance | Closure | Matched-size control | Closure |
| :-- | --: | --: | --: | :-- | --: |
| Initial disorder | 1 | 0.001 | 0.572 | | |
| Global | 3 | 0.040 | **0.918** | random-3 | 0.266 (± 0.06) |
| Macro-1 | 6 | 0.082 | 0.922 | random-6 | 0.449 |
| Macro-8 | 27 | 0.292 | 0.899 | random-27 | 0.653 |
| Regional-8 | 24 | 0.296 | 0.872 | | |
| Macro-32 | 99 | 0.470 | 0.823 | random-99 | 0.686 |
| Raw state | 128 | 0.460 | 0.706 | | |

Paired differences at τ = 12: global − random-3 = +0.652 [0.614, 0.689]; macro-8 − random-27 = +0.246 [0.230, 0.262]; global − initial disorder = +0.347 [0.339, 0.354]; global − raw = +0.212; all 10/10 seeds. With the future measured as the three global summaries, global closure is 0.929 and initial disorder 0.545.

**Horizon dependence.** Global closure falls slowly from 0.995 at τ = 1 to 0.918 at τ = 12. The initial-disorder scalar is almost perfectly closed at τ = 1 (0.997) and collapses to 0.572 by τ = 12: how much disorder there is predicts the next sweep, but how much remains after twelve sweeps depends on where the disorder sits relative to the unreliable region, which only the interaction feature carries.

**Convergence.** Low-dimensional matchings have converged (global 0.912 → 0.916 → 0.918 at 2,500 → 5,000 → 10,000 configurations); high-dimensional ones have not (macro-8 0.857 → 0.881 → 0.899; raw 0.622 → 0.666 → 0.706). The plain estimator is therefore a lower bound on closure for high-dimensional φ, and the ordering macro-8 < global, although macro-8 contains strictly more information, is a finite-sample matching artefact, not a property of the system. Fair comparisons are at matched size.

**Does microscopic detail add to the macrostate?** Matching on the three global summaries and then re-ranking the 50 candidates by raw-state distance raises closure by **+0.0103 [0.0094, 0.0112]** over the random re-rank control (10/10 seeds). Re-ranking by the previous sweep's global summaries raises it by only +0.0015 [0.0005, 0.0025]. At the macro-8 level, re-ranking by the *present* global summaries (+0.081) does as much as re-ranking by *history* (+0.072), and history is slightly worse than present-global (−0.009 [−0.009, −0.008]).

**Variance components at τ = 12:** V_total = 1.199 ± 0.025 positions², V_irr = 0.031 (both identical across abstractions).

## 5. Uncertainty

Across-seed SDs are 0.001–0.004 for every engineered abstraction and up to 0.06 for the 3-dimensional random projection. All paired differences reported above have intervals excluding zero and the same sign in every seed. The dominant uncertainty is not statistical but estimator bias: nearest-neighbour matching understates closure in high dimensions, by an amount the convergence table bounds from below.

## 6. Interpretation

- **The three global summaries are an approximately autonomous level of description for this target.** Configurations that agree on three numbers, matched almost exactly (relative neighbour distance 0.04), have futures twelve sweeps ahead whose true conditional means agree to within 8% of their total variance. This is a statement about the system, not about any learner.
- **The closure is not trivial sortedness.** The one-number disorder scalar closes at short horizons and fails at long ones; a random 3-dimensional projection closes at 27%. What closes is the specific combination of disorder, inversions and disorder-on-faults.
- **Microscopic detail adds a little, and the macrostate is approximately Markov.** Of the 8% of predictable variance not fixed by the three numbers, nearest raw-state neighbours recover about 1 point; the previous macrostate recovers essentially nothing. Because raw neighbours among 50 candidates are still far apart in 128 dimensions, the 1 point is a lower bound on what the microstate carries, and the 8% is an upper bound on it.
- **This answers the question Experiments 1–2 could not.** The usefulness of the supplied features is not only a coincidence between coordinates and a tree ensemble's inductive bias: the dynamics of this system, at this horizon and over this distribution of configurations, are largely captured by the three global variables. That is the premise of stage 2 of the research programme, and it holds here.
- **What it does not show.** No causal emergence (no interventions); no ontological claim; nothing about discovery (the variables were supplied); nothing beyond this simulator, this episode distribution, horizons ≤ 12 and the global target. The local target is not closed over any coarse representation (Experiments 1–2), and no test here changes that.

## 7. Limitations

- Nearest-neighbour matching, not exact macrostate cells: converged for ≤ 6 dimensions, biased downward above; a learner-free estimator that is unbiased in 128 dimensions does not exist at this sample size.
- Configurations are the episode distribution advanced by two sweeps; closure over a different distribution (for example, after many sweeps, or the shifted fault distribution) was not measured.
- The future is measured by the global summaries and the target; closure of a different macroscopic observable may differ.
- Realisation noise is corrected under the assumption of independent realisations, which holds by construction, and V_irr is small (3% of V_total at τ = 12), so noise correction is not the limiting factor.
- The history test uses one previous sweep of three global summaries; longer or richer histories were not tested.
