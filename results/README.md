# Results directory map

| Directory | What it is | Status |
| :-- | :-- | :-- |
| `baseline/` | Frozen first run (seed 20261106, 8 bins, fixed hyperparameters). Never modified. | Original, single seed |
| `manual-seed123-bins16/` | Copy of the manually executed `--seed 123 --bins 16` run. Not a controlled comparison with the baseline (seed and bins both changed). | Single seed |
| `experiment1/` | Robustness across 10 seeds, 4 budgets, 4 capacity presets, 5 representations, global and local tasks, with validation split, noise-floor estimate and cost accounting. | Multi-seed |
| `experiment2/` | Spatial granularity sweep (1–64 bins), feature-set ablations, dimension-matched random projections, size proxies, permutation importances and a granularity × capacity grid, on identical episodes per seed. | Multi-seed |
| `runs/` | Git-ignored scratch output of `uv run macro-prediction`. | Local only |

Each experiment directory contains:

- `raw/experiment<k>-seed-<seed>.jsonl`: one record per fitted model (and per size/importance/noise-floor measurement). This is the ground truth; everything else is derived from it.
- `metadata.json`: configuration, command line, software versions and platform of every invocation.
- `summary/*.csv`: across-seed statistics (means, SDs, 95% t-intervals, paired differences, validation-selected capacities, cost tables).
- `summary/tables.md`: Markdown tables regenerated from the CSVs; the numbers quoted in the documentation come from here.
- `figures/*.png|pdf`: publication figures.
- `README.md`: the narrative results summary (hypothesis, setup, metrics, results, uncertainty, interpretation, limitations).

Regenerate all derived files with:

```bash
uv run macro-analyze --experiment1 results/experiment1 --experiment2 results/experiment2
```
