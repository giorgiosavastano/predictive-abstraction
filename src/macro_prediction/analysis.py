"""Aggregate raw experiment records into tables, figures and Markdown summaries.

Reads the JSON-lines records written by ``macro-experiment1`` and
``macro-experiment2`` and writes, for each experiment directory:

``summary/*.csv``   machine-readable across-seed statistics,
``figures/*.png``   publication figures (also ``.pdf``),
``summary/tables.md`` the Markdown tables quoted in the documentation.

Every number in the documentation is taken from these files, which can be
regenerated at any time from the raw records with::

    uv run macro-analyze --experiment1 results/experiment1 --experiment2 results/experiment2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .models import CAPACITY_ORDER
from .stats import paired_difference, summarize
from .store import ResultStore

METRICS = ("test_rmse", "shift_rmse", "val_rmse", "test_r2")
COST_COLUMNS = ("feature_seconds_train", "feature_seconds_test", "fit_seconds", "predict_seconds",
                "total_leaves", "total_nodes", "n_trees", "n_features")


def _load(directory: Path, name: str) -> dict[str, pd.DataFrame]:
    records = ResultStore(directory, name).load_all()
    if not records:
        raise SystemExit(f"No records found under {directory}/raw")
    frame = pd.DataFrame(records)
    integer_columns = ("seed", "budget", "bins", "n_features", "n_trees", "total_leaves", "total_nodes",
                       "max_iter", "max_leaf_nodes", "predict_rows", "n_realizations",
                       "effective_rank_99", "n_constant_features")
    out = {}
    for kind in frame.record_type.unique():
        sub = frame[frame.record_type == kind].dropna(axis=1, how="all").copy()
        for col in integer_columns:
            # Columns are float in the combined frame because other record
            # types leave them empty; restore integer dtype per record type.
            if col in sub and sub[col].notna().all():
                sub[col] = sub[col].astype(int)
        out[kind] = sub
    return out


def _summarize_groups(fits: pd.DataFrame, group_cols: list[str], metrics=METRICS) -> pd.DataFrame:
    """Across-seed mean/SD/CI of each metric plus mean costs, per configuration."""
    rows = []
    for keys, block in fits.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        costs = {f"{c}_mean": float(block[c].mean()) for c in COST_COLUMNS if c in block}
        if {"feature_seconds_train", "fit_seconds", "predict_seconds", "feature_seconds_test"} <= set(block):
            costs["total_seconds_mean"] = float((block.feature_seconds_train + block.fit_seconds
                                                 + block.predict_seconds + block.feature_seconds_test).mean())
        for metric in metrics:
            if metric not in block:
                continue
            rows.append({**base, "metric": metric, **summarize(block[metric]), **costs,
                         "seeds": ",".join(str(s) for s in sorted(block.seed))})
    return pd.DataFrame(rows)


def _paired(fits: pd.DataFrame, group_cols: list[str], by: str, a: str, b: str,
            metric: str = "test_rmse") -> pd.DataFrame:
    """Paired across-seed comparison ``a - b`` of ``metric`` within each group."""
    rows = []
    for keys, block in fits.groupby(group_cols, sort=False):
        left = block[block[by] == a].set_index("seed")[metric]
        right = block[block[by] == b].set_index("seed")[metric]
        common = left.index.intersection(right.index)
        if len(common) < 2:
            continue
        rows.append({**dict(zip(group_cols, keys)), "metric": metric, "a": a, "b": b,
                     **paired_difference(left.loc[common], right.loc[common])})
    columns = [*group_cols, "metric", "a", "b", "n", "mean", "sd", "ci95_low", "ci95_high",
               "relative_mean_difference", "n_a_lower", "t_pvalue", "wilcoxon_pvalue"]
    return pd.DataFrame(rows, columns=columns)


def _md_table(frame: pd.DataFrame, columns: list[str], floatfmt: str = "{:.3f}") -> str:
    lines = ["| " + " | ".join(str(c) for c in columns) + " |", "|" + "|".join(" ---: " for _ in columns) + "|"]
    for i in range(len(frame)):
        cells = []
        for c in columns:
            v = frame[c].iloc[i]   # Column access keeps integer columns integer.
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "n/a")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ci(row) -> str:
    return f"{row['mean']:.3f} ± {row['sd']:.3f}"


# ------------------------------------------------------------- Experiment 1

def analyze_experiment1(directory: Path, figure_copy: Path | None, blog_dir: Path | None = None) -> None:
    tables = _load(directory, "experiment1")
    fits = tables["fit"]
    floor = tables.get("noise_floor", pd.DataFrame())
    out = directory / "summary"
    figs = directory / "figures"
    out.mkdir(exist_ok=True)
    fits.to_csv(out / "exp1_all_fits.csv", index=False)

    reps = ["Full state", "Macroscopic", "Random projection", "Initial disorder", "Full + macro"]
    capacities = [c for c in CAPACITY_ORDER if c in set(fits.capacity)]
    budgets = sorted(fits.budget.unique())
    seeds = sorted(fits.seed.unique())

    summary = _summarize_groups(fits, ["task", "capacity", "budget", "representation"])
    summary.to_csv(out / "exp1_summary.csv", index=False)

    pairs = []
    for a, b in (("Macroscopic", "Full state"), ("Macroscopic", "Random projection"),
                 ("Macroscopic", "Full + macro"), ("Macroscopic", "Initial disorder"),
                 ("Full state", "Initial disorder")):
        for metric in ("test_rmse", "shift_rmse"):
            pairs.append(_paired(fits, ["task", "capacity", "budget"], "representation", a, b, metric))
    paired = pd.concat(pairs, ignore_index=True)
    paired.to_csv(out / "exp1_paired.csv", index=False)

    # Validation-selected capacity: for every seed/task/budget/representation,
    # keep the capacity with the lowest *validation* RMSE, then report test RMSE.
    idx = fits.groupby(["seed", "task", "budget", "representation"])["val_rmse"].idxmin()
    selected = fits.loc[idx].copy()
    selected["capacity_selected"] = selected.capacity
    selected.to_csv(out / "exp1_validation_selected_fits.csv", index=False)
    selected_summary = _summarize_groups(selected, ["task", "budget", "representation"])
    choice_counts = (selected.groupby(["task", "budget", "representation"]).capacity
                     .value_counts().unstack(fill_value=0).reset_index())
    selected_summary = selected_summary.merge(choice_counts, on=["task", "budget", "representation"], how="left")
    selected_summary.to_csv(out / "exp1_validation_selected_summary.csv", index=False)
    selected_paired = pd.concat([
        _paired(selected, ["task", "budget"], "representation", "Macroscopic", "Full state", m)
        for m in ("test_rmse", "shift_rmse")] + [
        _paired(selected, ["task", "budget"], "representation", "Macroscopic", "Full + macro", "test_rmse"),
        _paired(selected, ["task", "budget"], "representation", "Full state", "Macroscopic", "test_rmse")],
        ignore_index=True)
    selected_paired.to_csv(out / "exp1_validation_selected_paired.csv", index=False)

    if not floor.empty:
        floor_summary = pd.DataFrame([{"task": t, **summarize(b.irreducible_rmse),
                                       "target_sd_mean": b.target_sd.mean(),
                                       "n_realizations": int(b.n_realizations.iloc[0])}
                                      for t, b in floor.groupby("task")])
        floor_summary.to_csv(out / "exp1_noise_floor.csv", index=False)

    cost = summary[summary.metric == "test_rmse"].rename(columns={"mean": "test_rmse_mean", "sd": "test_rmse_sd"})
    cost.to_csv(out / "exp1_cost.csv", index=False)

    # Figures.
    plots.exp1_learning_curves(summary, floor, figs / "exp1_learning_curves", capacities, reps)
    plots.exp1_learning_curves(summary, floor, figs / "exp1_learning_curves_shift", capacities, reps, metric="shift_rmse")
    if "local" in set(fits.task):
        plots.exp1_learning_curves(summary, floor, figs / "exp1_local_control",
                                   [c for c in capacities if c in set(fits[fits.task == "local"].capacity)],
                                   reps, task="local")
    plots.exp1_capacity(summary, floor, figs / "exp1_capacity", [budgets[0], budgets[-1]], reps, capacities)
    plots.exp1_cost(cost, figs / "exp1_cost_vs_error", reps, budgets[-1])
    if blog_dir is not None:
        plots.blog_learning_curve(selected_summary, floor, blog_dir / "blog_learning_curve")
        if "local" in set(fits.task):
            plots.blog_task_reversal(selected_summary, floor, blog_dir / "blog_task_reversal")

    # Markdown tables.
    md = [f"# Experiment 1 tables (auto-generated from `summary/*.csv`; {len(seeds)} seeds: {', '.join(map(str, seeds))})\n"]
    for task in ("global", "local"):
        if task not in set(summary.task):
            continue
        md.append(f"## Task: {task} — test RMSE, mean ± SD across seeds\n")
        for capacity in capacities:
            block = summary[(summary.task == task) & (summary.capacity == capacity) & (summary.metric == "test_rmse")]
            if block.empty:
                continue
            pivot = block.pivot(index="budget", columns="representation", values=["mean", "sd"])
            frame = pd.DataFrame({r: [f"{pivot['mean'][r][b]:.3f} ± {pivot['sd'][r][b]:.3f}" for b in pivot.index]
                                  for r in reps if r in pivot["mean"]}, index=pivot.index).reset_index()
            md.append(f"### Capacity `{capacity}`\n")
            md.append(_md_table(frame, list(frame.columns)) + "\n")
        sel = selected_summary[(selected_summary.task == task) & (selected_summary.metric == "test_rmse")]
        if not sel.empty:
            pivot = sel.pivot(index="budget", columns="representation", values=["mean", "sd"])
            frame = pd.DataFrame({r: [f"{pivot['mean'][r][b]:.3f} ± {pivot['sd'][r][b]:.3f}" for b in pivot.index]
                                  for r in reps if r in pivot["mean"]}, index=pivot.index).reset_index()
            md.append("### Validation-selected capacity (per seed, per representation)\n")
            md.append(_md_table(frame, list(frame.columns)) + "\n")
        pm = paired[(paired.task == task) & (paired.a == "Macroscopic") & (paired.b == "Full state") & (paired.metric == "test_rmse")]
        if not pm.empty:
            md.append("### Paired difference Macroscopic − Full state (test RMSE)\n")
            frame = pm[["capacity", "budget", "mean", "ci95_low", "ci95_high", "relative_mean_difference", "n_a_lower", "n", "t_pvalue", "wilcoxon_pvalue"]].copy()
            frame.columns = ["capacity", "budget", "mean diff", "CI low", "CI high", "relative", "seeds macro<full", "n seeds", "t p", "Wilcoxon p"]
            md.append(_md_table(frame, list(frame.columns), "{:.4f}") + "\n")
        sp = selected_paired[(selected_paired.task == task) & (selected_paired.a == "Macroscopic") & (selected_paired.b == "Full state") & (selected_paired.metric == "test_rmse")]
        if not sp.empty:
            md.append("### Paired difference Macroscopic − Full state, validation-selected capacities (test RMSE)\n")
            frame = sp[["budget", "mean", "ci95_low", "ci95_high", "relative_mean_difference", "n_a_lower", "n", "t_pvalue", "wilcoxon_pvalue"]].copy()
            frame.columns = ["budget", "mean diff", "CI low", "CI high", "relative", "seeds macro<full", "n seeds", "t p", "Wilcoxon p"]
            md.append(_md_table(frame, list(frame.columns), "{:.4f}") + "\n")
    if not floor.empty:
        md.append("## Irreducible error (Monte Carlo over future randomness, test episodes)\n")
        md.append(_md_table(floor_summary, ["task", "n", "mean", "sd", "target_sd_mean", "n_realizations"]) + "\n")
    md.append("## Costs at the largest budget (means over seeds; global task)\n")
    c = cost[(cost.task == "global") & (cost.budget == budgets[-1])][
        ["capacity", "representation", "n_features_mean", "test_rmse_mean", "feature_seconds_train_mean",
         "fit_seconds_mean", "predict_seconds_mean", "total_seconds_mean", "total_leaves_mean"]]
    md.append(_md_table(c, list(c.columns), "{:.4f}") + "\n")
    (out / "tables.md").write_text("\n".join(md))

    if figure_copy is not None:
        figure_copy.mkdir(parents=True, exist_ok=True)
        for png in figs.glob("exp1_*.png"):
            shutil.copy(png, figure_copy / png.name)
    print(f"Experiment 1 analysis written to {out} and {figs}")


# ------------------------------------------------------------- Experiment 2

def analyze_experiment2(directory: Path, figure_copy: Path | None, blog_dir: Path | None = None) -> None:
    tables = _load(directory, "experiment2")
    fits = tables["fit"]
    sizes = tables.get("size", pd.DataFrame())
    importance = tables.get("importance", pd.DataFrame())
    out = directory / "summary"
    figs = directory / "figures"
    out.mkdir(exist_ok=True)
    fits.to_csv(out / "exp2_all_fits.csv", index=False)

    group = ["task", "capacity", "budget", "feature_set", "bins", "edge_assignment"]
    summary = _summarize_groups(fits, group)
    if not sizes.empty:
        size_summary = sizes.groupby(["feature_set", "bins", "edge_assignment"]).agg(
            n_features=("n_features", "first"),
            marginal_entropy_bits_mean=("marginal_entropy_bits", "mean"),
            marginal_entropy_bits_sd=("marginal_entropy_bits", "std"),
            effective_rank_99_mean=("effective_rank_99", "mean"),
            mean_abs_correlation_mean=("mean_abs_correlation", "mean"),
            n_constant_features=("n_constant_features", "first"),
        ).reset_index()
        size_summary.to_csv(out / "exp2_size_metrics.csv", index=False)
        summary = summary.merge(size_summary.drop(columns=["n_features"]), on=["feature_set", "bins", "edge_assignment"], how="left")
    summary.to_csv(out / "exp2_summary.csv", index=False)

    main_capacity = fits[fits.task == "global"].groupby("capacity").budget.nunique().idxmax()
    budgets = sorted(fits.budget.unique())
    seeds = sorted(fits.seed.unique())
    capacities = [c for c in CAPACITY_ORDER if c in set(fits.capacity)]
    heat_budgets = sorted(fits[fits.capacity != main_capacity].budget.unique()) or [budgets[-1]]

    # Paired comparisons against the 8-bin baseline resolution and against global-only / raw.
    macro = fits[(fits.feature_set == "macro") & (fits.edge_assignment == "interior")].copy()
    macro["condition"] = macro.bins.astype(str)
    refs = fits[fits.feature_set.isin(["global", "raw"])].copy()
    refs["condition"] = refs.feature_set
    comp = pd.concat([macro, refs], ignore_index=True)
    pairs = []
    for cond in sorted(set(comp.condition)):
        if cond == "8":
            continue
        pairs.append(_paired(comp, ["task", "capacity", "budget"], "condition", cond, "8"))
    for cond in [str(b) for b in sorted(macro.bins.unique())]:
        pairs.append(_paired(comp, ["task", "capacity", "budget"], "condition", cond, "global"))
        pairs.append(_paired(comp, ["task", "capacity", "budget"], "condition", cond, "raw"))
    paired = pd.concat(pairs, ignore_index=True)
    paired.to_csv(out / "exp2_paired.csv", index=False)

    # Which resolution wins per seed (validation and test), macro feature set.
    best_rows = []
    for edge in sorted(set(macro.edge_assignment) | {"left"}):
        block = fits[(fits.feature_set == "macro") & (fits.edge_assignment == edge)]
        for keys, b in block.groupby(["task", "capacity", "budget", "seed"]):
            best_rows.append({"edge_assignment": edge, "task": keys[0], "capacity": keys[1], "budget": keys[2],
                              "seed": keys[3], "best_bins_val": int(b.loc[b.val_rmse.idxmin(), "bins"]),
                              "best_bins_test": int(b.loc[b.test_rmse.idxmin(), "bins"])})
    best = pd.DataFrame(best_rows)
    best.to_csv(out / "exp2_best_bins_per_seed.csv", index=False)
    best_counts = (best.groupby(["edge_assignment", "task", "capacity", "budget"]).best_bins_val
                   .value_counts().unstack(fill_value=0).reset_index())
    best_counts.to_csv(out / "exp2_best_bins_counts_validation.csv", index=False)

    # Permutation importances aggregated across seeds.
    imp_summary = pd.DataFrame()
    if not importance.empty:
        rows = []
        for _, r in importance.iterrows():
            for name, m, s in zip(r.feature_names, r.importances_mean, r.importances_std):
                rows.append({"seed": r.seed, "bins": r.bins, "feature": name, "importance": m, "within_seed_sd": s})
        long = pd.DataFrame(rows)
        imp_summary = long.groupby(["bins", "feature"]).agg(
            importance_mean=("importance", "mean"), importance_sd=("importance", "std"), n=("seed", "nunique")
        ).reset_index()
        imp_summary["group"] = np.where(imp_summary.feature.str.startswith("global"), "global", "regional")
        imp_summary["kind"] = imp_summary.feature.str.split("_", n=1).str[1]
        imp_summary.to_csv(out / "exp2_importance.csv", index=False)
        group_summary = long.assign(group=np.where(long.feature.str.startswith("global"), "global", "regional"),
                                    kind=long.feature.str.split("_", n=1).str[1])
        group_summary = group_summary.groupby(["bins", "group", "kind"]).importance.agg(["mean", "std", "sum"]).reset_index()
        group_summary.to_csv(out / "exp2_importance_groups.csv", index=False)

    # Figures.
    plots.exp2_error_vs_bins(summary, figs / "exp2_error_vs_bins", budgets, main_capacity)
    if "local" in set(fits.task):
        plots.exp2_error_vs_bins(summary, figs / "exp2_local_vs_bins", sorted(fits[fits.task == "local"].budget.unique()),
                                 main_capacity, task="local", feature_sets=("macro",))
    plots.exp2_error_vs_size(summary, figs / "exp2_error_vs_size", [budgets[0], budgets[-1]], main_capacity)
    plots.exp2_cost_vs_bins(summary, figs / "exp2_cost_vs_bins", budgets[-1], main_capacity)
    if len(capacities) > 1:
        plots.exp2_heatmap(summary, figs / "exp2_heatmap", heat_budgets, capacities)
    if not imp_summary.empty:
        plots.exp2_importance(imp_summary[imp_summary.bins == imp_summary.bins.min()], figs / "exp2_importance")
    if blog_dir is not None:
        plots.blog_granularity(summary, blog_dir / "blog_granularity", budgets[-1], main_capacity)

    # Markdown tables.
    md = [f"# Experiment 2 tables (auto-generated from `summary/*.csv`; {len(seeds)} seeds: {', '.join(map(str, seeds))})\n"]
    md.append(f"Main capacity: `{main_capacity}`. Edge rule `interior` is the original extractor (bins ≤ 32); `left` extends to 64 bins.\n")
    for task in ("global", "local"):
        block = summary[(summary.task == task) & (summary.capacity == main_capacity) & (summary.metric == "test_rmse")]
        if block.empty:
            continue
        md.append(f"## Task: {task} — test RMSE (mean ± SD across seeds), capacity `{main_capacity}`\n")
        for fs in ("macro", "regional", "random", "global", "raw"):
            for edge in ("interior", "left", "none"):
                b = block[(block.feature_set == fs) & (block.edge_assignment == edge)]
                if b.empty:
                    continue
                pivot = b.pivot(index="bins", columns="budget", values=["mean", "sd"])
                frame = pd.DataFrame({f"n={bud}": [f"{pivot['mean'][bud][k]:.3f} ± {pivot['sd'][bud][k]:.3f}" for k in pivot.index]
                                      for bud in pivot["mean"].columns}, index=pivot.index).reset_index()
                nf = b.groupby("bins").n_features_mean.first()
                frame.insert(1, "features", [int(nf[k]) for k in pivot.index])
                md.append(f"### `{fs}` ({edge})\n")
                md.append(_md_table(frame, list(frame.columns)) + "\n")
    if not sizes.empty:
        md.append("## Representation-size proxies (training pool, means over seeds)\n")
        s = size_summary[size_summary.edge_assignment.isin(["interior", "none"]) | (size_summary.bins == 64)]
        md.append(_md_table(s, ["feature_set", "bins", "edge_assignment", "n_features", "marginal_entropy_bits_mean",
                                "effective_rank_99_mean", "mean_abs_correlation_mean"], "{:.2f}") + "\n")
    pm = paired[(paired.task == "global") & (paired.capacity == main_capacity) & (paired.b == "8")]
    if not pm.empty:
        md.append("## Paired difference vs 8 bins (macro, interior rule, test RMSE, global task)\n")
        frame = pm[["budget", "a", "mean", "ci95_low", "ci95_high", "n_a_lower", "n", "t_pvalue"]].copy()
        frame.columns = ["budget", "condition", "mean diff", "CI low", "CI high", "seeds better than 8", "n seeds", "t p"]
        md.append(_md_table(frame.sort_values(["budget", "condition"]), list(frame.columns), "{:.4f}") + "\n")
    pg = paired[(paired.task == "global") & (paired.capacity == main_capacity) & (paired.b == "global")]
    if not pg.empty:
        md.append("## Paired difference macro(bins) − global-only (test RMSE, global task)\n")
        frame = pg[["budget", "a", "mean", "ci95_low", "ci95_high", "n_a_lower", "n", "t_pvalue"]].copy()
        frame.columns = ["budget", "bins", "mean diff", "CI low", "CI high", "seeds macro<global", "n seeds", "t p"]
        md.append(_md_table(frame.sort_values(["budget", "bins"]), list(frame.columns), "{:.4f}") + "\n")
    if not best.empty:
        md.append("## Best resolution per seed (validation RMSE), macro feature set\n")
        md.append(_md_table(best_counts, list(best_counts.columns), "{:.0f}") + "\n")
    if not imp_summary.empty:
        md.append("## Permutation importance by feature group (8 bins, sums over features, mean over seeds)\n")
        md.append(_md_table(group_summary, ["bins", "group", "kind", "mean", "std", "sum"], "{:.4f}") + "\n")
    if len(capacities) > 1:
        md.append("## Heatmap values: macro (left rule) test RMSE by bins × capacity\n")
        hm = summary[(summary.task == "global") & (summary.metric == "test_rmse") & (summary.feature_set == "macro")
                     & (summary.edge_assignment == "left") & (summary.budget.isin(heat_budgets))]
        for bud in heat_budgets:
            piv = hm[hm.budget == bud].pivot(index="bins", columns="capacity", values="mean").reindex(columns=capacities).reset_index()
            md.append(f"### {bud} training episodes\n")
            md.append(_md_table(piv, list(piv.columns)) + "\n")
    (out / "tables.md").write_text("\n".join(md))

    if figure_copy is not None:
        figure_copy.mkdir(parents=True, exist_ok=True)
        for png in figs.glob("exp2_*.png"):
            shutil.copy(png, figure_copy / png.name)
    print(f"Experiment 2 analysis written to {out} and {figs}")


# ------------------------------------------------------------ Experiment 2b

def analyze_experiment2b(directory: Path, figure_copy: Path | None, blog_dir: Path | None = None) -> None:
    tables = _load(directory, "experiment2b")
    rec = tables["closure"]
    for col in ("tau", "n_used", "n_features", "k", "n_configs", "n_realizations"):
        rec[col] = rec[col].astype(int)
    out = directory / "summary"
    figs = directory / "figures"
    out.mkdir(exist_ok=True)
    rec.to_csv(out / "exp2b_all_records.csv", index=False)
    seeds = sorted(rec.seed.unique())
    horizon = int(rec.tau.max())
    n_full = int(rec.n_used.max())

    group = ["representation", "base", "refined_by", "measure", "tau", "n_used"]
    rows = []
    for keys, block in rec.groupby(group, sort=False):
        base = dict(zip(group, keys))
        base["n_features"] = int(block.n_features.iloc[0])
        base["relative_neighbour_distance_mean"] = float(block.relative_neighbour_distance.mean())
        for metric in ("closure_index", "excess", "v_total", "v_irr"):
            rows.append({**base, "metric": metric, **summarize(block[metric])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "exp2b_summary.csv", index=False)
    closure = summary[summary.metric == "closure_index"]

    # Paired comparisons at the final horizon, full sample, target measure.
    full = rec[(rec.n_used == n_full) & (rec.measure == "target") & (rec.tau == horizon)]
    pairs = []
    for a, b in (("global", "random-3"), ("macro-8", "random-27"), ("macro-32", "random-99"),
                 ("global", "initial_disorder"), ("macro-8", "global"), ("macro-8", "regional-8"),
                 ("global", "raw"), ("macro-8", "raw"),
                 ("global+raw", "global+random"), ("global+history", "global+random"),
                 ("macro-8+raw", "macro-8+random"), ("macro-8+history", "macro-8+random"),
                 ("macro-8+global", "macro-8+random"), ("macro-8+history", "macro-8+global")):
        for metric in ("closure_index", "excess"):
            left = full[full.representation == a].set_index("seed")[metric]
            right = full[full.representation == b].set_index("seed")[metric]
            common = left.index.intersection(right.index)
            if len(common) >= 2:
                pairs.append({"a": a, "b": b, "metric": metric, "tau": horizon,
                              **paired_difference(left.loc[common], right.loc[common])})
    paired = pd.DataFrame(pairs)
    paired.to_csv(out / "exp2b_paired.csv", index=False)

    # Figures.
    main_reps = ["global", "macro-1", "macro-8", "macro-32", "regional-8", "random-3", "random-27", "raw",
                 "initial_disorder"]
    plots.exp2b_closure_vs_horizon(closure, figs / "exp2b_closure_vs_horizon", main_reps)
    plots.exp2b_closure_vs_horizon(closure, figs / "exp2b_closure_vs_horizon_global3", main_reps, measure="global3")
    plots.exp2b_convergence(closure, figs / "exp2b_convergence", ["global", "macro-8", "macro-32", "random-27", "raw"], horizon)
    plots.exp2b_refinements(closure, figs / "exp2b_refinements", horizon)
    if blog_dir is not None:
        plots.blog_closure(closure, blog_dir / "blog_closure")

    # Markdown tables.
    md = [f"# Experiment 2b tables (auto-generated from `summary/*.csv`; {len(seeds)} seeds: {', '.join(map(str, seeds))})\n",
          f"Configurations per seed: {n_full:,}; realisations per configuration: {int(rec.n_realizations.iloc[0])}; "
          f"k = {int(rec.k.iloc[0])} matched neighbours; horizon {horizon} sweeps.\n"]
    for measure in ("target", "global3"):
        c = closure[(closure.measure == measure) & (closure.n_used == n_full) & (closure.refined_by == "none")]
        piv = c.pivot(index="representation", columns="tau", values="mean")
        feats = c.groupby("representation").n_features.first()
        dist = c.groupby("representation").relative_neighbour_distance_mean.first()
        cols = [t for t in dict.fromkeys((1, 2, 4, 8, horizon)) if t in piv.columns]
        frame = piv[cols].copy()
        frame.insert(0, "rel. nbr dist", [dist[r] for r in frame.index])
        frame.insert(0, "features", [int(feats[r]) for r in frame.index])
        frame = frame.reset_index().rename(columns={t: f"τ={t}" for t in cols})
        md.append(f"## Closure index (mean over seeds), future measured as `{measure}`, full sample\n")
        md.append(_md_table(frame, list(frame.columns)) + "\n")
    c = closure[(closure.measure == "target") & (closure.tau == horizon) & (closure.refined_by == "none")]
    piv = c.pivot(index="representation", columns="n_used", values="mean").reset_index()
    md.append(f"## Convergence: closure index at τ={horizon} vs configurations used (target)\n")
    md.append(_md_table(piv, list(piv.columns)) + "\n")
    c = closure[(closure.measure == "target") & (closure.tau == horizon) & (closure.n_used == n_full)
                & (closure.representation.str.contains(r"\+|^global$|^macro-8$"))]
    frame = c[["representation", "mean", "sd", "ci95_low", "ci95_high", "relative_neighbour_distance_mean"]]
    md.append(f"## Refinements at τ={horizon} (target): closure after re-ranking 50 candidates\n")
    md.append(_md_table(frame, list(frame.columns)) + "\n")
    frame = paired[paired.metric == "closure_index"][["a", "b", "mean", "ci95_low", "ci95_high", "n_a_lower", "n", "t_pvalue"]].copy()
    frame.columns = ["a", "b", "mean diff (closure)", "CI low", "CI high", "seeds a<b", "n seeds", "t p"]
    md.append(f"## Paired differences in closure index at τ={horizon} (target)\n")
    md.append(_md_table(frame, list(frame.columns), "{:.4f}") + "\n")
    v = summary[(summary.metric.isin(["v_total", "v_irr"])) & (summary.measure == "target") & (summary.representation == "global")
                & (summary.n_used == n_full)][["tau", "metric", "mean", "sd"]]
    md.append("## Variance components of the target by horizon (positions²; identical for every representation)\n")
    md.append(_md_table(v, list(v.columns), "{:.4f}") + "\n")
    (out / "tables.md").write_text("\n".join(md))

    if figure_copy is not None:
        figure_copy.mkdir(parents=True, exist_ok=True)
        for png in figs.glob("exp2b_*.png"):
            shutil.copy(png, figure_copy / png.name)
    print(f"Experiment 2b analysis written to {out} and {figs}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment1", type=Path, default=None)
    parser.add_argument("--experiment2", type=Path, default=None)
    parser.add_argument("--experiment2b", type=Path, default=None)
    parser.add_argument("--paper-figures", type=Path, default=Path("paper/figures"),
                        help="Directory receiving copies of the PNG figures (pass '' to skip).")
    parser.add_argument("--blog-figures", type=Path, default=Path(""),
                        help="Directory receiving the single-panel blog figures (e.g. docs/figures); off by default.")
    args = parser.parse_args(argv)
    copy = args.paper_figures if str(args.paper_figures) else None
    blog = args.blog_figures if str(args.blog_figures) else None
    if args.experiment1 is None and args.experiment2 is None and args.experiment2b is None:
        parser.error("Provide --experiment1, --experiment2 and/or --experiment2b")
    if args.experiment1 is not None:
        analyze_experiment1(args.experiment1, copy, blog)
    if args.experiment2 is not None:
        analyze_experiment2(args.experiment2, copy, blog)
    if args.experiment2b is not None:
        analyze_experiment2b(args.experiment2b, copy, blog)


if __name__ == "__main__":
    main()
