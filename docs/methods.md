# Methods

This document describes the simulator, representations, learners and evaluation protocol for the original baseline and for the two multi-seed experiments. The code in `src/macro_prediction/` is the authoritative specification; this is a readable summary with pointers.

## 1. Hypothesis and estimand

**Task:** predict a future *global* observable from an initial snapshot of a stochastic local-sorting system. We measure out-of-sample squared-error prediction performance under fixed training-sample budgets and fixed learner capacities. The hypothesis is that a hand-engineered coarse representation offers a better inductive bias for this target than raw coordinates or random compression, *for a finite-resource learner*.

Because the macro features are deterministic functions of the full state, a Bayes-optimal prediction based on the full state cannot be worse than a Bayes-optimal prediction based on the macro state. All comparisons here involve *specific finite-data learners*, not Bayes-optimal predictors.

## 2. Simulator (`simulator.py`)

- Array length `N = 64`, a permutation of integers 0…63; episodes begin sorted.
- For each episode, shuffle one contiguous block (75% probability) or two independently selected contiguous blocks (25%). Block widths are integers from 5 through 32.
- Edge reliability is 0.90 at each exchange edge. One fault interval receives a single uniform reliability from `[0.15, 0.65)` (main distribution) or `[0.05, 0.30)` (shifted distribution). Fault widths are 5…20 positions (main) or 5…30 (shifted); see `make_episodes` for the exact `integers` bounds. The 64th reliability slot is padding (always 1.0).
- Each of 12 sweeps performs an even-edge phase and an odd-edge phase. Each phase considers disjoint adjacent pairs; an inverted pair swaps with the edge's reliability probability.
- **Global target:** mean absolute positional displacement after 12 sweeps, `mean_i |a[i] - i|`, in positions per element.
- **Local control target:** the value at position 32 after one full sweep.
- **Irreducible error (new):** because future exchange attempts are random, the target has conditional variance given the snapshot. `conditional_target_statistics` re-simulates each test episode 32 times with fresh randomness; the root of the mean conditional variance is the RMSE that no snapshot-based predictor can beat in expectation. It is reported alongside every learning curve.

## 3. Representations (`features.py`)

| Name | Columns | Definition |
| :-- | --: | :-- |
| Full state (raw) | 128 | `a / 63` (64 values) and the 64 edge reliabilities |
| Global summaries | 3 | mean normalized displacement; descent fraction over the 63 edges; mean of displacement × (1 − reliability) over the 63 edges |
| Regional summaries | 3 × bins | per region: mean displacement, descent fraction, mean unreliability |
| Macroscopic | 3 + 3 × bins | global + regional (the baseline uses 8 bins → 27) |
| Initial disorder | 1 | the first global summary only |
| Random projection | 3 + 3 × bins | standardized raw state × fixed Gaussian matrix (entries scaled by `1/sqrt(k)`) |
| Full + macro | 128 + 3 + 3 × bins | diagnostic concatenation |

Regions are `numpy.array_split(arange(64), bins)`. Two rules assign edges to regions:

- `interior` (original extractor): only edges fully inside the region (`inds[:-1]`). Boundary edges are seen by no region. Single-element regions have no edges, so the rule is defined for 1 ≤ bins ≤ 32.
- `left` (new): edge `e` belongs to the region containing position `e`; every edge is counted exactly once and 64 bins are defined (the last region's two edge features are constant zero). With one bin both rules coincide exactly (test-enforced).

What is **preserved** at every resolution: the three global summaries (if included) and, per region, how far elements are from home on average, how many local inversions there are and how unreliable the exchanges are. What is **discarded** at every resolution, including 64 bins: element identities, the sign of displacements and the exact arrangement of elements within a region. The 64-bin representation is therefore *not* the raw state; it is a per-element magnitude-of-displacement map plus per-edge descent and unreliability indicators.

Representation-size proxies (`representation_size_metrics`): feature count; a *marginal-entropy encoding bound* (each column discretized into ≤255 quantile bins, as the learner does internally, entropies summed in bits; an upper bound on the joint entropy of what the learner sees, not a mutual information with the target); effective rank (principal components for 99% variance); mean absolute inter-feature correlation.

## 4. Learner and capacity presets (`models.py`)

`HistGradientBoostingRegressor` with `learning_rate = 0.075`, `min_samples_leaf = 12`, `l2_regularization = 2.0`, `early_stopping = False`, `random_state = seed`, and:

| Preset | Trees | Max leaves per tree | Note |
| :-- | --: | --: | :-- |
| small | 50 | 4 | |
| medium | 100 | 9 | identical to the baseline |
| large | 300 | 31 | |
| xlarge | 1,000 | 63 | Experiment 1, 5,000-example budget only (cost: ~20–40 s per fit) |

The realised size of every fitted ensemble (trees, total leaves, total nodes) is recorded, because the number of leaves actually grown depends on the data. Fitting cost, inference cost on 2,000 episodes and feature-extraction cost (median of five repeats for the training subset and for the test split) are recorded per fit. Timings are wall-clock seconds on one machine (Apple Silicon, 10 cores, scikit-learn's default OpenMP threading) and are indicative, not benchmarks.

## 5. Data splits and random streams (`data.py`)

Every stream is `numpy.random.default_rng(seed + offset)`:

| Offset | Stream | Used for |
| --: | :-- | :-- |
| 0 | train | training-pool episodes and their trajectories |
| 1 | test | in-distribution test episodes and trajectories |
| 2 | shift | shifted-fault test episodes and trajectories |
| 3 | project | random-projection matrix |
| 4 | val | validation episodes and trajectories (new) |
| 10 / 11 / 12 | local | local-task trajectories for train / test / val |
| 20 | floor | repeated trajectories for the noise-floor estimate (new) |
| 30 / 31 / 32 | closure | Experiment 2b configurations, their futures, and random projections (new) |

Offsets 0–3 and 10–11 are the baseline's; with `seed = 20261106`, 8 bins and the `medium` preset the new pipeline reproduces every baseline number at the 5,000 budget exactly (verified; see §8). Experiment seeds are 1000, 2000, …, 10000, at least 100 apart so that derived streams never overlap (enforced by `validate_seed_spacing`).

Per seed: a 5,000-episode training pool, 1,000 validation episodes, 2,000 in-distribution test episodes and 2,000 shifted test episodes. Training budgets are nested prefixes of the pool, so every representation and capacity sees exactly the same episodes at a given budget. The pool size must stay fixed within an output directory (the trajectories depend on it); the runners refuse to resume into a directory produced with a different pool, validation or test size.

**Validation discipline.** Hyperparameter choice (the capacity preset) is made on validation RMSE only, per seed, budget and representation; the test and shifted splits are never used for selection. The random projection's standardizer is fitted on the budget's training subset only (a correction to the baseline, which fitted it on the whole 5,000-episode pool; this changed the baseline's random-projection numbers at budgets below 5,000 by at most ±0.03 RMSE and nothing else).

## 6. Experiment 1 protocol (`experiment1.py`)

Grid: 10 seeds × budgets {250, 750, 2,000, 5,000} × presets {small, medium, large} × 5 representations × tasks {global, local}, plus `xlarge` at the 5,000 budget on the global task (local task at `xlarge` not run). Metrics: validation, test and shifted RMSE and R²; costs and model sizes as above; noise floor per seed (32 realizations).

Analysis (`analysis.py`): per-configuration means, SDs and 95% t-intervals across seeds; paired across-seed differences (macro − full state, macro − random projection, macro − full + macro, macro − initial disorder, full − initial disorder) with t-interval, paired t-test, Wilcoxon signed-rank and the number of seeds in which the first representation is better; a *validation-selected* comparison in which each representation uses, per seed and budget, the preset with the lowest validation RMSE; cost-versus-error tables.

## 7. Experiment 2 protocol (`experiment2.py`)

Same episodes within each seed for every condition. Conditions: `macro`, `regional` and `random` at bins ∈ {1, 2, 4, 8, 16, 32, 64} (the `interior` rule stops at 32), plus `global` (3 features) and `raw` (128) references. Main sweep at the `medium` preset for every budget; `small` and `large` presets at budgets 250 and 5,000 for the granularity × capacity grid; local task for `macro` at every resolution and budget (`medium`). Representation-size proxies on the 5,000-episode pool; permutation importance (validation split, 10 repeats, RMSE scoring) for the 8-bin macro model at 5,000.

## 7b. Experiment 2b protocol (`experiment2b.py`)

Learner-free closure test. Per seed: 10,000 episodes advanced by 2 sweeps (stream `+30`), each simulated forward 16 times (stream `+31`) recording the three global summaries after every sweep to τ = 12. For each candidate abstraction φ (initial disorder, global, macro at 1–32 bins, regional-8, random projections of 3/6/27/99 dimensions seeded from stream `+32`, raw), each configuration is matched to its k = 10 nearest distinct configurations in standardized φ-space. With μ_i the realisation mean and s²_i the realisation variance of a future measure (target, or the three global summaries scaled by their SDs), excess = mean over pairs of (μ_i − μ_j)²/2 − mean(s²)/16, V_total = Var(μ) − mean(s²)/16, closure = 1 − excess / V_total. The estimator is recomputed at 2,500 and 5,000 configurations (convergence), and refinements re-rank the k0 = 50 nearest in φ by raw-state distance, by the previous sweep's global summaries, by the present global summaries, or at random (control). Distinctness is judged on the full microstate (values and reliabilities). A distance-extrapolated estimator was tried and rejected because neighbour distances concentrate in high dimensions.

## 8. Reproducibility checks performed

1. `uv run macro-prediction` after the refactor reproduces `results/baseline/*.csv` exactly (all metric fields; timings excluded).
2. `macro-experiment1 --seeds 20261106 --capacities medium` reproduces all 20 baseline rows at the 5,000 budget and all rows except the random projection at smaller budgets (expected, §5).
3. Tests (`uv run pytest`) check the refactored extractor against a verbatim copy of the original code for every bin count, the edge-assignment invariants, split determinism and disjointness, result-store resumability and the end-to-end CLIs.

## 9. Known limitations of the protocol

- One learner family (histogram gradient boosting) and one hyperparameter axis (trees × leaves); learning rate and regularization are fixed. A locality-aware neural predictor for the raw state remains untested (future work).
- Timings are single-machine wall-clock numbers under default multithreading; they order configurations but are not portable benchmarks.
- The shifted split changes only the fault distribution, not the dynamics, array size or permutation class.
- The noise-floor estimate uses 32 realizations per episode; its own Monte Carlo error is small relative to the gaps reported but not zero.
- Representation-size proxies are bounds and redundancy measures, not mutual-information estimates.
- The macro features embed knowledge of the sorted target configuration; these experiments test the usefulness of *supplied* abstractions, not autonomous abstraction discovery.
