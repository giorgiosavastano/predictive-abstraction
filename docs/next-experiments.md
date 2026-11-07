# Next experiments — status and remaining directions

The original pre-registered plan listed three experiments. The first two have now been executed (see `results/experiment1/README.md` and `results/experiment2/README.md`); the third remains future work.

## 1. Replication and strong baselines — **executed (Experiment 1)**

**Question:** Is the engineered-summary advantage robust, or a seed / learner artifact?

**Done:** 10 seeds; budgets 250–5,000; four capacity presets of the gradient-boosting learner; validation-based capacity selection; paired across-seed intervals; a dimension-matched random projection re-drawn per seed; noise-floor estimation; feature-extraction, fit and inference timing; realised model sizes.

**Outcome:** The advantage replicates in 10/10 seeds at every budget and capacity, but shrinks from ~45–49% at 250 examples to 17% at 5,000 examples with validation-selected capacity. The local control inverts the ranking in 10/10 seeds.

**Not done (still open):**

- Compare against locality-aware raw-state architectures (small 1-D CNN / MLP) with a fair tuning budget. The planned falsification "the macro advantage disappears when the full-state model uses a locality-aware architecture" has not been tested.
- Sweep learning rate / early stopping for very large boosting models; the xlarge preset over-fits under the fixed learning rate, so the raw state's best achievable error with this learner family is not yet pinned down.
- Larger budgets (≥ 20,000 episodes) to see where the raw-state learning curve meets the macro curve.

## 2. Spatial scale / compression sweep — **executed (Experiment 2)**

**Question:** Is there an intermediate spatial scale that is optimal for global recovery prediction?

**Done:** 1–64 bins on identical episodes per seed; macro / regional-only / global-only ablations; matched-size random projections; representation-size proxies; permutation importances; granularity × capacity grid; local control at every resolution.

**Outcome:** A shallow interior optimum (8 bins at medium capacity, 8–16 at large) that disappears at 250 examples and moves finer with resources. The pronounced optimum is in the regional-only ablation (4–8 bins). Most of the benefit comes from three engineered global statistics, especially displacement × unreliability; even 64-bin engineered features beat the raw state by 26%.

**Not done (still open):**

- Vary the prediction horizon (currently 12 sweeps) to test whether the optimal scale depends on how far ahead one predicts.
- Replace the hand-engineered interaction feature with its components to see whether the learner can recover it from `global_displacement` and a global unreliability mean alone.
- Mutual-information or predictive-information estimates for the low-dimensional representations (feasible at 3–6 features, not at 128).

## 2b. Emergent predictive structure — **executed (Experiment 2b)**

**Question:** Do microscopic configurations with the same macroscopic representation have the same macroscopic future? That is, do the sorting dynamics approximately close over the macro variables (P(Z_{t+τ} | X_t) ≈ P(Z_{t+τ} | Z_t)), or are the variables merely convenient coordinates for one learner?

**Done:** 10 seeds × 10,000 configurations × 16 futures, nearest-neighbour matching in each candidate abstraction, horizons 1–12, matched-size random projections and the disorder scalar as controls, convergence check at three sample sizes, and controlled refinements (raw detail, previous macrostate, present macrostate, random).

**Outcome:** Not falsified. Closure at τ = 12 is 0.918 ± 0.004 for the three global summaries (0.27 for a matched random projection, 0.57 for the disorder scalar); raw detail recovers ≈ 1 point more, the previous macrostate ≈ 0. The estimator is a lower bound above six dimensions.

**Not done (still open):** other configuration distributions (late recovery, shifted faults); longer horizons; deeper histories; other array lengths and disturbance types; the local target (expected to fail); a dimension-fair closure estimator.

## 3. Discover abstractions instead of providing them — **not started (future work)**

**Question:** Can a model learn representations that improve resource-bounded prediction and resemble interpretable coarse variables?

- Train a small encoder `z = f_theta(x)` and predictor `y_hat = g_phi(z)` end-to-end on the same episodes. Use a fixed bottleneck size (e.g. 3, 8, 16, 27 dimensions). Compare against an autoencoder trained for reconstruction and a random encoder of the same size.
- To prevent the unconstrained encoder simply memorizing inputs, limit representation capacity and quantify it (noise injection, regularization, discretization, or variational information-bottleneck approximations). **Latent dimension alone is not a rigorous measure of information bits**, as the entropy bounds in Experiment 2 illustrate.
- Evaluate whether `z` supports multiple related future tasks, and whether probes recover initial displacement, disorder and fault overlap *without those values being supplied during training*. Given Experiment 2, the sharpest test is whether a learned 3-dimensional bottleneck recovers something like the displacement × unreliability interaction.
- Include the local prediction task as a negative control.

**Possible falsification:** Learned low-dimensional representations show no consistent efficiency benefit over direct raw-state predictors; apparent compression is not actually reducing effective information content.

**Smallest test (from the manuscript, §11.2):** at 5,000 episodes, does a learned 3-dimensional bottleneck reach the global-only error (0.405 ± 0.009), and do linear probes recover the displacement × unreliability interaction from it?

## 4. Abstraction transfer — **not started (future work)**

**Question:** Does a representation learned in one family of sorting environments reduce the data needed in a structurally related, unseen family?

- Source: the present simulator. Targets: different array length, horizon, local rule (e.g. insertion-style exchanges) or fault geometry. Compare frozen and fine-tuned encoders against learning from scratch at every budget.
- Controls: a surface-feature control (relabel positions so attribute overlap is destroyed but relations survive), a negative-transfer control (a target family where the source abstraction misleads), and full accounting of pre-training cost.
- **Smallest test:** with 250 target episodes, does the transferred encoder reach the error that learning from scratch needs 2,000 episodes for, and does the surface-feature control preserve the gain? Success across seeds of one simulator does not count as transfer.

## 5. Adaptive abstraction — **not started (future work)**

**Question:** How efficiently can a system recognize that a previously adequate abstraction is insufficient for a new objective, and revise it?

- Train on the global target, then switch to the local target (or a different horizon) without announcing the switch. Give the system a representation menu (macro at several resolutions, raw, union) with a validation-driven selector or learned gate.
- Measure episodes needed to detect degradation and reach a representation within 5% of the best available; compare with retraining from scratch. Controls: a non-adapting system (39% worse on the local target per Experiment 1) and a switch to a task where the old abstraction still works (false-alarm rate).
- **Smallest test:** if detection plus re-selection costs more episodes than retraining from scratch, adaptation is not cheaper than relearning here.

## Publication gate — current status

| Criterion | Status |
| :-- | :-- |
| Pre-declared primary target and metric | met (global displacement, test RMSE) |
| Independent test data held out from all selection | met (validation split added) |
| ≥ 10 seeds, paired intervals, negative controls | met (10 seeds; 20 originally proposed) |
| Learner-free evidence that the useful variables are a level of description of the system | met for the global target (Experiment 2b); not for other targets or systems |
| Strong architecture-aware raw-state baselines | **not met** |
| Dimension-matched baselines | met (random projections at every size) |
| Computational budget accounting | met (timings + model sizes; single machine) |
| Public code, lockfile, reproducible figures, command lines | met |
| Literature review beyond the motivating references | partially met (12 verified references; no systematic review) |

The repository now supports a narrow, well-controlled claim about *supplied* task-relevant representations. It does not yet support claims about learned abstractions, and the strongest raw-state competitor has not been tried.
