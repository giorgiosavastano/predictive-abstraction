# Experiment 2b — Emergent predictive structure (approximate closure test)

All numbers are copied from `summary/tables.md` and `summary/*.csv`, regenerated from `raw/*.jsonl` by `uv run macro-analyze --experiment2b results/experiment2b`. Mean ± SD across 10 seeds; paired 95% t-intervals. No learner is involved anywhere in this experiment.

## 1. Hypothesis

Experiments 1–2 showed that a few supplied collective variables are *useful to a learner*. Here we ask a narrower system-level question: do configurations with similar summaries have similar average futures, without fitting a predictor? Full observational predictive sufficiency would require agreement of entire future distributions, P(Z_{t+τ} | X_t) ≈ P(Z_{t+τ} | Z_t). The present diagnostic tests only conditional means of the measured observables. We call high agreement “approximate conditional-mean closure”; it does not establish full distributional sufficiency, a closed macroscopic transition law, Markovian dynamics or causal emergence.

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

**Horizon dependence.** Global closure falls slowly from 0.995 at τ = 1 to 0.918 at τ = 12. The initial-disorder scalar gives high conditional-mean agreement at τ = 1 (0.997), falling to 0.572 by τ = 12. This is consistent with a role for the location of disorder relative to unreliable exchanges; the matching test does not identify a unique necessary feature or a causal mechanism.

**Convergence.** The global estimate changes less with sample size (0.912 → 0.916 → 0.918 at 2,500 → 5,000 → 10,000 configurations) than the higher-dimensional estimates (macro-8 0.857 → 0.881 → 0.899; raw 0.622 → 0.666 → 0.706). This is evidence of downward matching bias at higher dimensions, not a proof of convergence at low dimensions or a rigorous lower bound. Macro-8 contains the global summaries, so its lower estimated score is consistent with imperfect matching rather than an information loss. Matched-size controls reduce the dimensional confound but do not remove differences in geometry.

**Does microscopic detail add to the macrostate?** Matching on the three global summaries and then re-ranking the 50 candidates by raw-state distance raises closure by **+0.0103 [0.0094, 0.0112]** over the random re-rank control (10/10 seeds). Re-ranking by the previous sweep's global summaries raises it by only +0.0015 [0.0005, 0.0025]. At the macro-8 level, re-ranking by the *present* global summaries (+0.081) does as much as re-ranking by *history* (+0.072), and history is slightly worse than present-global (−0.009 [−0.009, −0.008]).

**Variance components at τ = 12:** V_total = 1.199 ± 0.025 positions², V_irr = 0.031 (both identical across abstractions).

## 5. Uncertainty

Across-seed SDs are 0.001–0.004 for every engineered abstraction and up to 0.06 for the 3-dimensional random projection. The paired differences reported above have intervals excluding zero and the same sign in every seed. Those intervals describe variation across seeds, not uncertainty from the matching approximation. The convergence checks indicate substantial estimator bias in higher dimensions; they do not bound that bias rigorously.

## 6. Interpretation

- **Evidence for approximate conditional-mean closure.** The three supplied global summaries give a high score for the measured average recovery, over the sampled distribution and horizons. A score of 0.918 leaves about 8% of the noise-corrected conditional-mean variance as residual disagreement in this estimator. It does not mean that individual futures or full future distributions agree to 92% accuracy.
- **Content matters.** The disorder scalar and matched-size random projection give lower scores at twelve sweeps. This supports the usefulness of the supplied combination, without identifying a unique sufficient representation.
- **Some refinement information remains.** Raw-state re-ranking improves closure by 0.0103 and one-step-history re-ranking by 0.0015 relative to random re-ranking. The history improvement is small but nonzero. Neither refinement measures all the information in the omitted variables; imperfect matching remains. These results do not establish a Markov property.
- **Separate evidence about the system and the learner.** The average-future regularity is observed without fitting a learner. It complements the learning results, but does not exclude a contribution from the tree learner’s inductive bias or prove a general autonomous macroscopic law.
- **Scope.** No interventions or autonomous representation learning were performed. The local task in Experiments 1–2 tests learning performance; local-target closure has not been measured.

## 7. Limitations

- Nearest-neighbour matching approximates equal-summary cells. The sample-size checks suggest relatively stable low-dimensional estimates and downward bias in higher dimensions; they provide no rigorous estimator bounds.
- Configurations are sampled after two recovery sweeps; other configuration distributions were not tested. Horizons are one to twelve further sweeps.
- The future is measured by average displacement and the three standardized global summaries. Agreement of those conditional means does not imply agreement of future distributions or complete trajectories.
- Noise correction assumes independent realisations, generated with independent draws by construction. V_irr is about 3% of V_total at τ = 12; finite-realisation and matching error remain.
- Only one previous sweep’s three global summaries were used to refine matches. Longer and richer histories, and full conditional distributions, would be needed to investigate a Markov description more fully.
