# Manual run: seed 123, 16 spatial bins

These files are a verbatim copy of `results/runs/experiment-123/`, which was produced manually on 2026-11-06 with the original single-run CLI:

```bash
uv run macro-prediction --seed 123 --bins 16 --output results/runs/experiment-123
```

`results/runs/` is git-ignored, so the copy here preserves the run alongside the checked-in baseline.

## What the run contains

- `run.json`: seed 123, 64 elements, 12 sweeps, **16 bins**, budgets 250/750/2,000/5,000, 2,000 test episodes; NumPy 2.3.5, scikit-learn 1.8.0, Matplotlib 3.10.8.
- `learning_curves.csv`: all 20 rows (5 representations × 4 budgets) with in-distribution and shifted RMSE/R². The run completed normally.
- `local_control.csv`: the 3-row local-task control.
- `learning_curves.png`: the single-run figure.

## Is it comparable to the baseline?

**No, not as a controlled comparison.** Two things changed at once relative to `results/baseline/` (seed 20261106, 8 bins): the random seed, which regenerates every training, test and shifted episode, every future trajectory and the random-projection matrix; and the number of spatial bins, which changes the macro representation from 27 to 51 features and the random projection from 27 to 51 dimensions. Any difference between the two runs mixes seed-to-seed variation with the effect of the bin count.

For reference, the test RMSE at 5,000 training episodes was:

| Representation | Baseline (seed 20261106, 8 bins) | Manual run (seed 123, 16 bins) |
| :-- | --: | --: |
| Full state | 0.50188 | 0.52248 |
| Macroscopic | 0.36727 | 0.37193 |
| Random projection | 0.70781 | 0.63505 |
| Initial disorder | 0.72810 | 0.74265 |
| Full + macro | 0.37770 | 0.38180 |

Qualitatively the ordering is the same (macro < full + macro < full state < random projection ≈ initial disorder), and the local control again favours the full state (RMSE 3.378 vs 4.541 for macro). The controlled version of the bin-count question is Experiment 2 (`results/experiment2/`), which varies the bin count on identical episodes within each of ten seeds; the controlled version of the seed question is Experiment 1 (`results/experiment1/`).
