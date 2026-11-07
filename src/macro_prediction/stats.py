"""Across-seed summary statistics and paired comparisons."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def summarize(values) -> dict:
    """Mean, SD and a 95% t-interval for the mean across independent seeds."""
    x = np.asarray(list(values), dtype=np.float64)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        low, high = mean - half, mean + half
    else:
        low = high = float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": float(low), "ci95_high": float(high)}


def paired_difference(a, b) -> dict:
    """Paired statistics for ``a - b`` across seeds (same seed in both).

    Reports the mean difference with a 95% t-interval, the paired t-test
    p-value, the Wilcoxon signed-rank p-value (robust to non-normality, but
    coarse for ten seeds), and how many seeds have ``a < b``. The relative
    difference is ``mean(a - b) / mean(b)``.
    """
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have equal length")
    diff = a - b
    out = summarize(diff)
    out["relative_mean_difference"] = float(diff.mean() / b.mean()) if b.mean() else float("nan")
    out["n_a_lower"] = int((diff < 0).sum())
    if len(diff) > 1 and np.any(diff != 0):
        out["t_pvalue"] = float(sps.ttest_rel(a, b).pvalue)
        out["wilcoxon_pvalue"] = float(sps.wilcoxon(a, b).pvalue)
    else:
        out["t_pvalue"] = out["wilcoxon_pvalue"] = float("nan")
    return out
