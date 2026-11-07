# Frozen baseline provenance

These files are copies of the first experiment run performed on 2026-11-06, using the original monolithic `abstraction_sorting_experiment.py`. They have not been rounded again or reconstructed from text.

- `learning_curves.csv`: all five model conditions at four training-data budgets, with held-out, shifted-test, R², and machine-local model-fit times.
- `local_control.csv`: local center-value prediction results for three models.
- `learning_curves.png`: the original saved figure.

Original simulation settings: 64 elements; 12 full even/odd sweeps; 8 spatial bins; seed 20261106; 5,000 training episodes; 2,000 in-distribution and 2,000 shifted test episodes. Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0 and Matplotlib 3.10.8.

After modularizing the source, we ran the default CLI anew and compared numerical CSV fields. **All 20 rows of global predictive metrics and all three rows of local predictive metrics matched exactly** in the verification environment. The `fit_seconds` column was excluded from that equality check since timing depends on execution environment.

These files are a **single-seed preliminary baseline**, not results averaged across repeated trials. Do not report them as robust experimental evidence without the follow-up replications.
