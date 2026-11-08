"""Publication figures for Experiments 1 and 2.

Conventions: one fixed colour per representation (identity never depends on
which series happen to be on a panel), thin lines, shaded 95% confidence
bands across seeds, a single y-axis per panel, recessive solid grid lines and
no top/right spines. Figures are written as PNG (300 dpi) and PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    # Experiment 1 representations.
    "Full state": "#2a78d6",
    "Macroscopic": "#eb6834",
    "Random projection": "#1baf7a",
    "Initial disorder": "#eda100",
    "Full + macro": "#e87ba4",
    # Experiment 2 feature sets.
    "raw": "#2a78d6",
    "macro": "#eb6834",
    "random": "#1baf7a",
    "global": "#eda100",
    "regional": "#6250d6",
    "noise floor": "#6b6b6b",
}
FEATURE_SET_LABELS = {
    "macro": "Global + regional (macro)",
    "regional": "Regional only",
    "global": "Global only (3 features)",
    "random": "Random projection (matched size)",
    "raw": "Full state (128 features)",
}
CAPACITY_LABELS = {"small": "small\n50×4", "medium": "medium\n100×9", "large": "large\n300×31", "xlarge": "xlarge\n1000×63"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "figure.dpi": 120, "savefig.dpi": 300,
})


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _band(ax, x, mean, low, high, color, label, marker="o", ls="-"):
    ax.plot(x, mean, marker=marker, color=color, label=label, ls=ls)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)


# --------------------------------------------------------------------- Exp 1

def exp1_learning_curves(summary: pd.DataFrame, floor: pd.DataFrame | None, path: Path,
                         capacities: list[str], representations: list[str], task: str = "global",
                         metric: str = "test_rmse") -> None:
    """RMSE vs training budget, one panel per capacity, bands = 95% CI across seeds."""
    sub = summary[(summary.task == task) & (summary.metric == metric)]
    # A learning curve needs at least two budgets; presets run at one budget
    # only (e.g. xlarge at 5,000) appear in the capacity figure instead.
    capacities = [c for c in capacities if sub[sub.capacity == c].budget.nunique() >= 2]
    fig, axes = plt.subplots(1, len(capacities), figsize=(3.4 * len(capacities), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, capacity in zip(axes, capacities):
        block = sub[sub.capacity == capacity]
        for name in representations:
            rows = block[block.representation == name].sort_values("budget")
            if rows.empty:
                continue
            _band(ax, rows.budget, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[name], name)
        if floor is not None and not floor.empty:
            f = floor[floor.task == task]["irreducible_rmse"].mean()
            ax.axhline(f, color=COLORS["noise floor"], ls="--", lw=1, label="irreducible error (est.)")
        ax.set_xscale("log")
        budgets = sorted(block.budget.unique())
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(f"capacity: {CAPACITY_LABELS.get(capacity, capacity).replace(chr(10), ' ')}")
        ax.set_xlabel("Training episodes")
    axes[0].set_ylabel({"test_rmse": "Test RMSE (positions)", "shift_rmse": "Shifted-test RMSE (positions)"}.get(metric, metric))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.08))
    label = "global remaining displacement" if task == "global" else "local centre value after one sweep"
    fig.suptitle(f"Experiment 1 — {label}: error vs training budget (mean ± 95% CI over seeds)", y=1.02)
    save(fig, path)


def exp1_capacity(summary: pd.DataFrame, floor: pd.DataFrame | None, path: Path, budgets: list[int],
                  representations: list[str], capacities: list[str], task: str = "global") -> None:
    """RMSE vs capacity preset, one panel per budget."""
    sub = summary[(summary.task == task) & (summary.metric == "test_rmse")]
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, budget in zip(axes, budgets):
        block = sub[sub.budget == budget]
        for name in representations:
            rows = block[block.representation == name].set_index("capacity").reindex(capacities).dropna(subset=["mean"])
            if rows.empty:
                continue
            x = [capacities.index(c) for c in rows.index]
            _band(ax, x, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[name], name)
        if floor is not None and not floor.empty:
            ax.axhline(floor[floor.task == task]["irreducible_rmse"].mean(), color=COLORS["noise floor"],
                       ls="--", lw=1, label="irreducible error (est.)")
        ax.set_xticks(range(len(capacities)))
        ax.set_xticklabels([CAPACITY_LABELS[c] for c in capacities])
        ax.set_title(f"{budget:,} training episodes")
        ax.set_xlabel("Model capacity (trees × max leaves)")
    axes[0].set_ylabel("Test RMSE (positions)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24))
    fig.suptitle("Experiment 1 — error vs model capacity (mean ± 95% CI over seeds)", y=1.02)
    save(fig, path)


def exp1_cost(cost: pd.DataFrame, path: Path, representations: list[str], budget: int) -> None:
    """Total wall-clock cost vs test RMSE; markers = capacities, lines join one representation."""
    sub = cost[(cost.task == "global") & (cost.budget == budget)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, xcol, xlabel in zip(axes, ("total_seconds_mean", "total_leaves_mean"),
                                ("Wall-clock seconds (features + fit + inference)", "Total leaves in fitted ensemble")):
        for name in representations:
            rows = sub[sub.representation == name].sort_values(xcol)
            if rows.empty:
                continue
            ax.plot(rows[xcol], rows.test_rmse_mean, marker="o", color=COLORS[name], label=name, lw=1.2)
            if name in ("Full state", "Macroscopic"):   # Letters on the two main series only.
                for _, r in rows.iterrows():
                    ax.annotate(r.capacity[0].upper() if r.capacity != "xlarge" else "XL", (r[xcol], r.test_rmse_mean),
                                textcoords="offset points", xytext=(4, 3 if name == "Full state" else -9),
                                fontsize=7, color="#52514e")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Test RMSE (positions)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"Experiment 1 — cost vs error at {budget:,} training episodes (means over seeds; S/M/L/XL = capacity preset)", y=1.02)
    save(fig, path)


# --------------------------------------------------------------------- Exp 2

def exp2_error_vs_bins(summary: pd.DataFrame, path: Path, budgets: list[int], capacity: str,
                       task: str = "global", feature_sets=("macro", "regional", "random")) -> None:
    """RMSE vs number of spatial bins, one panel per budget; references for global-only and raw."""
    sub = summary[(summary.task == task) & (summary.capacity == capacity) & (summary.metric == "test_rmse")]
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, budget in zip(axes, budgets):
        block = sub[sub.budget == budget]
        for fs in feature_sets:
            for edge, ls, marker in (("interior", "-", "o"), ("left", ":", "s"), ("none", "-", "o")):
                rows = block[(block.feature_set == fs) & (block.edge_assignment == edge)].sort_values("bins")
                if rows.empty:
                    continue
                label = FEATURE_SET_LABELS[fs] + ("" if edge in ("interior", "none") else " [left-edge rule]")
                _band(ax, rows.bins, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[fs], label, marker, ls)
        for fs in ("global", "raw"):
            rows = block[block.feature_set == fs]
            if rows.empty:
                continue
            ax.axhline(rows["mean"].iloc[0], color=COLORS[fs], lw=1.2, label=FEATURE_SET_LABELS[fs])
            ax.axhspan(rows.ci95_low.iloc[0], rows.ci95_high.iloc[0], color=COLORS[fs], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        bins = sorted(block.bins.unique())
        bins = [b for b in bins if b > 0]
        ax.set_xticks(bins)
        ax.set_xticklabels([str(b) for b in bins])
        ax.set_title(f"{budget:,} training episodes")
        ax.set_xlabel("Spatial bins")
    axes[0].set_ylabel("Test RMSE (positions)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    label = "global remaining displacement" if task == "global" else "local centre value after one sweep"
    fig.suptitle(f"Experiment 2 — {label}: error vs spatial granularity ({capacity} capacity; mean ± 95% CI)", y=1.02)
    save(fig, path)


def exp2_error_vs_size(summary: pd.DataFrame, path: Path, budgets: list[int], capacity: str) -> None:
    """RMSE vs representation size: feature count (left) and marginal-entropy bound (right)."""
    sub = summary[(summary.task == "global") & (summary.capacity == capacity) & (summary.metric == "test_rmse")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for ax, xcol, xlabel in zip(axes, ("n_features_mean", "marginal_entropy_bits_mean"),
                                ("Number of input features", "Marginal-entropy encoding bound (bits)")):
        for budget, alpha in zip(budgets, np.linspace(0.45, 1.0, len(budgets))):
            block = sub[sub.budget == budget]
            for fs, marker in (("macro", "o"), ("regional", "^"), ("random", "s")):
                rows = block[(block.feature_set == fs) & (block.edge_assignment.isin(["interior", "none"]))].sort_values(xcol)
                if rows.empty:
                    continue
                ax.plot(rows[xcol], rows["mean"], marker=marker, color=COLORS[fs], alpha=alpha,
                        label=f"{FEATURE_SET_LABELS[fs]}, n={budget:,}" if ax is axes[0] else None)
            for fs, marker in (("global", "D"), ("raw", "X")):
                rows = block[block.feature_set == fs]
                if not rows.empty:
                    ax.plot(rows[xcol], rows["mean"], marker=marker, color=COLORS[fs], alpha=alpha, ls="none",
                            label=f"{FEATURE_SET_LABELS[fs]}, n={budget:,}" if ax is axes[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Test RMSE (positions)")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=1)
    fig.suptitle(f"Experiment 2 — error vs representation size ({capacity} capacity; lighter = smaller budget)", y=1.02)
    save(fig, path)


def exp2_cost_vs_bins(summary: pd.DataFrame, path: Path, budget: int, capacity: str) -> None:
    sub = summary[(summary.task == "global") & (summary.capacity == capacity) & (summary.budget == budget)
                  & (summary.metric == "test_rmse") & (summary.feature_set.isin(["macro", "random"]))
                  & (summary.edge_assignment.isin(["interior", "none"]))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    cols = [("feature_seconds_train_mean", "Feature extraction (train, s)"),
            ("fit_seconds_mean", "Model fit (s)"), ("predict_seconds_mean", "Inference on 2,000 episodes (s)")]
    for ax, (col, label) in zip(axes, cols):
        for fs in ("macro", "random"):
            rows = sub[sub.feature_set == fs].sort_values("bins")
            ax.plot(rows.bins, rows[col], marker="o", color=COLORS[fs], label=FEATURE_SET_LABELS[fs])
        raw = summary[(summary.task == "global") & (summary.capacity == capacity) & (summary.budget == budget)
                      & (summary.metric == "test_rmse") & (summary.feature_set == "raw")]
        if not raw.empty:
            ax.axhline(raw[col].iloc[0], color=COLORS["raw"], lw=1.2, label=FEATURE_SET_LABELS["raw"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        bins = sorted(sub.bins.unique())
        ax.set_xticks(bins)
        ax.set_xticklabels([str(b) for b in bins])
        ax.set_xlabel("Spatial bins")
        ax.set_title(label)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Experiment 2 — computational cost vs granularity ({budget:,} episodes, {capacity} capacity; means over seeds)", y=1.03)
    save(fig, path)


def exp2_heatmap(summary: pd.DataFrame, path: Path, budgets: list[int], capacities: list[str],
                 feature_set: str = "macro") -> None:
    """Test RMSE across granularity (rows) and capacity (columns), one panel per budget."""
    sub = summary[(summary.task == "global") & (summary.metric == "test_rmse") & (summary.feature_set == feature_set)
                  & (summary.edge_assignment == "left")]
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets), 3.6))
    axes = np.atleast_1d(axes)
    vmin, vmax = sub["mean"].min(), sub["mean"].max()
    for ax, budget in zip(axes, budgets):
        block = sub[sub.budget == budget].pivot(index="bins", columns="capacity", values="mean").reindex(columns=capacities)
        image = ax.imshow(block.values, cmap="Blues_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(capacities)))
        ax.set_xticklabels([CAPACITY_LABELS[c] for c in capacities])
        ax.set_yticks(range(len(block.index)))
        ax.set_yticklabels([str(b) for b in block.index])
        ax.set_ylabel("Spatial bins")
        ax.set_title(f"{budget:,} training episodes")
        ax.grid(False)
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                v = block.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if v < (vmin + vmax) / 2 else "#0b0b0b")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Test RMSE (positions), mean over seeds")
    fig.suptitle("Experiment 2 — macro representation: error across granularity and capacity", y=1.02)
    save(fig, path)


# -------------------------------------------------------------------- Exp 2b

CLOSURE_COLORS = {
    "global": "#eda100", "macro-8": "#eb6834", "macro-1": "#f0a37a", "macro-32": "#b5461c",
    "regional-8": "#6250d6", "random-3": "#8fd9bc", "random-27": "#1baf7a", "random-99": "#117a54",
    "raw": "#2a78d6", "initial_disorder": "#9c9a94",
}


def exp2b_closure_vs_horizon(summary: pd.DataFrame, path: Path, representations: list[str],
                             measure: str = "target") -> None:
    """Closure index vs prediction horizon, full sample, one line per representation."""
    sub = summary[(summary.measure == measure) & (summary.n_used == summary.n_used.max())
                  & (summary.refined_by == "none")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name in representations:
        rows = sub[sub.representation == name].sort_values("tau")
        if rows.empty:
            continue
        color = CLOSURE_COLORS.get(name, "#52514e")
        _band(ax, rows.tau, rows["mean"], rows.ci95_low, rows.ci95_high, color,
              f"{name} ({int(rows.n_features.iloc[0])})")
    ax.set_xlabel("Prediction horizon τ (sweeps ahead)")
    ax.set_ylabel("Closure index  1 − excess / V_total")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(sub.tau.unique()))
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    label = "global target (mean displacement)" if measure == "target" else "three global summaries"
    fig.suptitle(f"Experiment 2b — how much of the predictable future is fixed by the macrostate? ({label}; mean ± 95% CI)",
                 y=1.01, fontsize=10)
    save(fig, path)


def exp2b_convergence(summary: pd.DataFrame, path: Path, representations: list[str], tau: int,
                      measure: str = "target") -> None:
    """Closure index vs number of configurations used (estimator convergence check)."""
    sub = summary[(summary.measure == measure) & (summary.tau == tau) & (summary.refined_by == "none")]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name in representations:
        rows = sub[sub.representation == name].sort_values("n_used")
        if rows.empty:
            continue
        color = CLOSURE_COLORS.get(name, "#52514e")
        _band(ax, rows.n_used, rows["mean"], rows.ci95_low, rows.ci95_high, color, name)
    ax.set_xscale("log", base=2)
    sizes = sorted(sub.n_used.unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Configurations used for matching")
    ax.set_ylabel(f"Closure index at τ = {tau}")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Experiment 2b — estimator convergence: low-dimensional matchings converge, high-dimensional ones do not",
                 y=1.01, fontsize=10)
    save(fig, path)


def exp2b_refinements(summary: pd.DataFrame, path: Path, tau: int, measure: str = "target") -> None:
    """Closure after re-ranking the macro-neighbourhood by raw, history, present global, or at random."""
    sub = summary[(summary.measure == measure) & (summary.tau == tau)
                  & (summary.n_used == summary.n_used.max())]
    groups = [("global", ["global", "global+random", "global+raw", "global+history"]),
              ("macro-8", ["macro-8", "macro-8+random", "macro-8+raw", "macro-8+global", "macro-8+history"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for ax, (base, names) in zip(axes, groups):
        rows = sub.set_index("representation").reindex(names).dropna(subset=["mean"])
        labels = [n.replace(base, "").lstrip("+") or "k nearest" for n in rows.index]
        colors = ["#52514e" if "random" in n else CLOSURE_COLORS.get(base, "#52514e") for n in rows.index]
        x = np.arange(len(rows))
        ax.errorbar(x, rows["mean"], yerr=[rows["mean"] - rows.ci95_low, rows.ci95_high - rows["mean"]],
                    fmt="none", ecolor="#0b0b0b", elinewidth=0.9, capsize=3)
        ax.scatter(x, rows["mean"], c=colors, s=60, zorder=3)
        control = rows["mean"].get(f"{base}+random")
        if control is not None:
            ax.axhline(control, color="#52514e", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"match on {base}, then re-rank 50 candidates by…", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Closure index at τ = {tau} (axis does not start at 0)")
    lo = sub[sub.representation.isin(sum((n for _, n in groups), []))]["ci95_low"].min()
    hi = sub[sub.representation.isin(sum((n for _, n in groups), []))]["ci95_high"].max()
    axes[0].set_ylim(lo - 0.01, hi + 0.01)
    fig.suptitle("Experiment 2b — does microscopic detail or history add to the macrostate? "
                 "(grey point/line = random re-rank control; mean ± 95% CI over seeds)", y=1.02, fontsize=10)
    save(fig, path)


# ------------------------------------------------------------- Blog figures
# Single-panel versions with larger type for a web page. They read the same
# summary tables as the paper figures, so the numbers cannot drift.

BLOG_RC = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
           "lines.linewidth": 2.0, "lines.markersize": 6}
BLOG_REPRESENTATION_LABELS = {
    "Full state": "Full state",
    "Macroscopic": "Summary",
    "Random projection": "Random combinations",
    "Initial disorder": "Initial disorder",
}
BLOG_FEATURE_SET_LABELS = {
    "macro": "Global + regional (summary)",
    "regional": "Regional only (summary − 3 global numbers)",
    "random": "Random combinations (same size as full summary)",
}


def _direct_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color,
                fontsize=10, va="center", ha="left")


def blog_learning_curve(selected: pd.DataFrame, floor: pd.DataFrame | None, path: Path) -> None:
    """Global task, validation-selected capacity: error vs budget, four representations."""
    with plt.rc_context(BLOG_RC):
        sub = selected[(selected.task == "global") & (selected.metric == "test_rmse")]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name in ("Full state", "Random projection", "Initial disorder", "Macroscopic"):
            rows = sub[sub.representation == name].sort_values("budget")
            if rows.empty:
                continue
            label = BLOG_REPRESENTATION_LABELS[name]
            _band(ax, rows.budget, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[name], label)
            _direct_label(ax, rows.budget.iloc[-1], rows["mean"].iloc[-1], label, COLORS[name])
        if floor is not None and not floor.empty:
            f = floor[floor.task == "global"]["irreducible_rmse"].mean()
            ax.axhline(f, color=COLORS["noise floor"], ls="--", lw=1.2)
            _direct_label(ax, sub.budget.max(), f, "Estimated\nerror floor", COLORS["noise floor"], dy=16)
        budgets = sorted(sub.budget.unique())
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b:,}" for b in budgets])
        ax.set_xlim(budgets[0] * 0.9, budgets[-1] * 2.2)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Test error (RMSE, positions per element)")
        ax.set_title("Predicting remaining disorder\n"
                     "(model size selected on validation data; mean ± 95% CI over 10 seeds)", loc="left")
        ax.set_ylim(bottom=0)
        save(fig, path)


def blog_task_reversal(selected: pd.DataFrame, floor: pd.DataFrame | None, path: Path) -> None:
    """Two panels: the global task (macro wins) and the local task (full state wins)."""
    with plt.rc_context(BLOG_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        fig.subplots_adjust(top=0.8, wspace=0.32)
        titles = {"global": "Global question:\nremaining disorder after 12 rounds",
                  "local": "Local question:\ncentral value after 1 round"}
        for ax, task in zip(axes, ("global", "local")):
            sub = selected[(selected.task == task) & (selected.metric == "test_rmse")]
            for name in ("Full state", "Macroscopic", "Initial disorder"):
                rows = sub[sub.representation == name].sort_values("budget")
                if rows.empty:
                    continue
                _band(ax, rows.budget, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[name],
                      BLOG_REPRESENTATION_LABELS[name])
            if floor is not None and not floor.empty:
                ax.axhline(floor[floor.task == task]["irreducible_rmse"].mean(), color=COLORS["noise floor"],
                           ls="--", lw=1.2, label="Estimated error floor")
            budgets = sorted(sub.budget.unique())
            ax.set_xscale("log")
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{b:,}" for b in budgets])
            ax.set_xlabel("Training examples")
            ax.set_title(titles[task], loc="left", fontsize=11)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel("Test error (RMSE, positions per element)")
        axes[1].set_ylabel("Test error (RMSE, element values)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("The same summary on two different prediction questions\n"
                     "(model size selected on validation data; mean ± 95% CI over seeds)",
                     y=1.02, fontsize=12)
        save(fig, path)


def blog_granularity(summary: pd.DataFrame, path: Path, budget: int, capacity: str) -> None:
    """Error vs regions, with a second axis showing full-summary input counts."""
    with plt.rc_context(BLOG_RC):
        sub = summary[(summary.task == "global") & (summary.capacity == capacity) & (summary.budget == budget)
                      & (summary.metric == "test_rmse")]
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        for fs in ("random", "regional", "macro"):
            rows = sub[(sub.feature_set == fs) & (sub.edge_assignment.isin(["left", "none"]))].sort_values("bins")
            if rows.empty:
                continue
            _band(ax, rows.bins, rows["mean"], rows.ci95_low, rows.ci95_high, COLORS[fs],
                  BLOG_FEATURE_SET_LABELS[fs])
        for fs, text in (("raw", "Full state, 128 numbers (reference)"), ("global", "3 global numbers only (reference)")):
            rows = sub[sub.feature_set == fs]
            if rows.empty:
                continue
            ax.axhline(rows["mean"].iloc[0], color=COLORS[fs], lw=1.4, label=text)
        bins = sorted(b for b in sub.bins.unique() if b > 0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(bins)
        ax.set_xticklabels([str(b) for b in bins])
        counts_ax = ax.secondary_xaxis(
            "top", functions=(lambda regions: 3 + 3 * regions, lambda counts: (counts - 3) / 3))
        counts = [3 + 3 * b for b in bins]
        counts_ax.set_xticks(counts)
        counts_ax.set_xticklabels([str(count) for count in counts])
        counts_ax.set_xlabel("Input numbers per example (full summary and random combinations)", labelpad=8)
        counts_ax.tick_params(axis="x", labelsize=10)
        counts_ax.spines["top"].set_visible(True)
        counts_ax.spines["top"].set_color("#b7b7b2")
        counts_ax.grid(False)
        ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelspacing=0.7)
        ax.set_xlabel("Number of regions the array is divided into")
        ax.set_ylabel("Test error (RMSE, positions per element)")
        ax.set_title("How much detail to keep?\n"
                     f"({budget:,} training examples; {capacity} model; mean ± 95% CI over seeds)",
                     loc="left", fontsize=11, pad=54)
        ax.set_ylim(bottom=0)
        save(fig, path)


def blog_closure(summary: pd.DataFrame, path: Path) -> None:
    """Closure vs horizon for the blog: global, disorder scalar, random-3, raw."""
    with plt.rc_context(BLOG_RC):
        sub = summary[(summary.measure == "target") & (summary.n_used == summary.n_used.max())
                      & (summary.refined_by == "none")]
        labels = {"global": "3 global numbers (summary)", "initial_disorder": "Initial disorder (1 number)",
                  "random-3": "Random combinations (3 numbers)",
                  "raw": "Full state (128 numbers;\nlower-bound estimate)"}
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name in ("global", "initial_disorder", "raw", "random-3"):
            rows = sub[sub.representation == name].sort_values("tau")
            if rows.empty:
                continue
            color = CLOSURE_COLORS.get(name, "#52514e")
            _band(ax, rows.tau, rows["mean"], rows.ci95_low, rows.ci95_high, color, labels[name])
            _direct_label(ax, rows.tau.iloc[-1], rows["mean"].iloc[-1], labels[name], color)
        ax.set_xlim(0.5, 19)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Rounds of sorting ahead")
        ax.set_ylabel("Closure index (1 = agreement of average futures)")
        ax.set_title("Similar descriptions: how similar are the average futures?\n"
                     "(mean ± 95% CI over 10 seeds)",
                     loc="left")
        save(fig, path)


def exp2_importance(importance: pd.DataFrame, path: Path) -> None:
    """Permutation importance (validation RMSE increase) per feature, mean ± SD over seeds."""
    rows = importance.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 0.22 * len(rows) + 1.2))
    colors = [COLORS["global"] if n.startswith("global") else COLORS["regional"] for n in rows.feature]
    ax.barh(rows.feature, rows.importance_mean, xerr=rows.importance_sd, color=colors, height=0.7,
            error_kw={"elinewidth": 0.8, "ecolor": "#52514e"})
    ax.set_xlabel("Increase in validation RMSE when the feature is permuted (positions)")
    ax.set_title("Experiment 2 — permutation importance, 8-bin macro model, 5,000 episodes (mean ± SD over seeds)")
    ax.grid(axis="y", visible=False)
    save(fig, path)
