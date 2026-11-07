"""Experiment 2: spatial granularity and information loss.

For every seed the *same* episodes (training pool, validation, test, shift)
are used for every spatial resolution and every feature set, so that
differences between granularities reflect the representation only.

Conditions
----------
``feature_set`` is one of:

``macro``     global summaries + regional summaries (the existing extractor);
``regional``  regional summaries only;
``global``    the three global summaries only (independent of ``bins``);
``random``    a dimension-matched random projection with ``3 + 3 * bins`` columns;
``raw``       the full 128-dimensional microscopic state (reference).

``edge_assignment`` is ``interior`` (original extractor, ``bins <= 32``) or
``left`` (every edge belongs to the region containing its left position,
which makes 64 bins well defined). Both series are run so that the 64-bin
point can be placed on a series that is internally consistent.

The main sweep uses one capacity preset at every budget. A smaller grid over
capacities (``--heatmap-capacities``) at selected budgets supports the
granularity-by-capacity heatmap. The local control task is evaluated for the
``macro`` feature set at every resolution. Representation-size proxies
(feature count, marginal-entropy encoding bound, effective rank, mean
absolute correlation) are computed on the training pool once per seed, and
permutation feature importances are computed on the validation split for the
baseline 8-bin macro model.

Example::

    uv run macro-experiment2 --output results/experiment2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

from .data import DEFAULT_SEEDS, SeedData, STREAM_OFFSETS, generate_seed_data, validate_seed_spacing
from .features import (EDGE_ASSIGNMENTS, GLOBAL_DIM, N, REGIONAL_PER_BIN, RandomProjection,
                       macro_feature_names, macro_features, raw_features, representation_size_metrics,
                       validate_bins)
from .models import CAPACITY_ORDER, EvalSet, fit_and_evaluate
from .store import ResultStore

DEFAULT_BINS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_BUDGETS = (250, 750, 2000, 5000)
FEATURE_SETS = ("macro", "regional", "global", "random", "raw")
SPLITS = ("train", "val", "test", "shift")


def _rows(data: SeedData, budget: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "train": (data.train.arrays[:budget], data.train.success[:budget]),
        "val": (data.val.arrays, data.val.success),
        "test": (data.test.arrays, data.test.success),
        "shift": (data.shift.arrays, data.shift.success),
    }


def _timed(fn, *args, repeats: int = 5):
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def condition_features(data: SeedData, budget: int, feature_set: str, bins: int,
                       edge_assignment: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Features for all splits of one condition, plus extraction cost (train subset / test)."""
    rows = _rows(data, budget)
    feats, cost = {}, {}
    if feature_set == "raw":
        for split, (x, p) in rows.items():
            feats[split], t = _timed(raw_features, x, p)
            if split in ("train", "test"):
                cost[f"feature_seconds_{split}"] = t
    elif feature_set == "random":
        n_components = GLOBAL_DIM + REGIONAL_PER_BIN * bins
        raw = {s: raw_features(x, p) for s, (x, p) in rows.items()}
        _, t_raw_train = _timed(raw_features, *rows["train"])
        _, t_raw_test = _timed(raw_features, *rows["test"])
        projection, t_fit = _timed(
            lambda: RandomProjection(n_components, seed + STREAM_OFFSETS["project"]).fit(raw["train"]))
        for split in rows:
            feats[split], t = _timed(projection.transform, raw[split])
            if split == "train":
                cost["feature_seconds_train"] = t_raw_train + t_fit + t
            elif split == "test":
                cost["feature_seconds_test"] = t_raw_test + t
    else:
        kwargs = dict(bins=bins, include_global=feature_set in ("macro", "global"),
                      include_regional=feature_set in ("macro", "regional"),
                      edge_assignment=edge_assignment)
        for split, (x, p) in rows.items():
            feats[split], t = _timed(lambda x_, p_: macro_features(x_, p_, **kwargs), x, p)
            if split in ("train", "test"):
                cost[f"feature_seconds_{split}"] = t
    return feats, cost


def iter_conditions(bins_list, edge_assignments):
    """Yield ``(feature_set, bins, edge_assignment)`` for every valid condition."""
    yield ("raw", 0, "none")
    yield ("global", 0, "none")
    for bins in bins_list:
        yield ("random", bins, "none")
        for edge in edge_assignments:
            if (edge == "interior" and bins > N // 2):
                continue
            yield ("macro", bins, edge)
            yield ("regional", bins, edge)


def run_seed(seed: int, store: ResultStore, *, bins_list, budgets, capacity, heatmap_capacities,
             heatmap_budgets, edge_assignments, n_val, n_test, importance_bins, local_budgets) -> None:
    existing = store.load(seed)
    data = generate_seed_data(seed, n_train=max(budgets), n_val=n_val, n_test=n_test, with_local=True)
    conditions = list(iter_conditions(bins_list, edge_assignments))

    # Representation-size proxies on the full training pool (no targets involved).
    for feature_set, bins, edge in conditions:
        key = f"size|{feature_set}|{bins}|{edge}"
        if key in existing:
            continue
        feats, _ = condition_features(data, max(budgets), feature_set, bins, edge, seed)
        record = {"key": key, "record_type": "size", "seed": seed, "feature_set": feature_set,
                  "bins": bins, "edge_assignment": edge,
                  **representation_size_metrics(feats["train"])}
        store.append(seed, record)
        existing[key] = record

    # Fits. The capacity grid is: main capacity at every budget; heatmap
    # capacities at the heatmap budgets only.
    plan: list[tuple[int, str]] = [(b, capacity) for b in budgets]
    for b in heatmap_budgets:
        for c in heatmap_capacities:
            if (b, c) not in plan:
                plan.append((b, c))

    for budget, cap in plan:
        for feature_set, bins, edge in conditions:
            tasks = ["global"]
            if feature_set == "macro" and cap == capacity and budget in local_budgets:
                tasks.append("local")
            for task in tasks:
                key = f"{task}|{budget}|{cap}|{feature_set}|{bins}|{edge}"
                if key in existing:
                    continue
                feats, cost = condition_features(data, budget, feature_set, bins, edge, seed)
                if task == "global":
                    y_train = data.train.target[:budget]
                    eval_sets = [EvalSet("val", feats["val"], data.val.target),
                                 EvalSet("test", feats["test"], data.test.target),
                                 EvalSet("shift", feats["shift"], data.shift.target)]
                else:
                    y_train = data.train.local_target[:budget]
                    eval_sets = [EvalSet("val", feats["val"], data.val.local_target),
                                 EvalSet("test", feats["test"], data.test.local_target)]
                record = fit_and_evaluate(cap, seed, feats["train"], y_train, eval_sets)
                model = record.pop("_model")
                record.update({"key": key, "record_type": "fit", "seed": seed, "task": task,
                               "budget": budget, "feature_set": feature_set, "bins": bins,
                               "edge_assignment": edge, **cost})
                store.append(seed, record)
                existing[key] = record
                print(f"seed={seed} {task:6s} n={budget:5d} {cap:6s} {feature_set:8s} bins={bins:2d} "
                      f"{edge:8s} val={record['val_rmse']:.3f} test={record['test_rmse']:.3f}", flush=True)

                # Permutation importance for the reference macro model, on validation data.
                if (task == "global" and feature_set == "macro" and edge == "interior"
                        and bins in importance_bins and cap == capacity and budget == max(budgets)):
                    ikey = f"importance|{budget}|{cap}|{bins}|{edge}"
                    if ikey not in existing:
                        result = permutation_importance(
                            model, feats["val"], data.val.target, n_repeats=10,
                            random_state=seed, scoring="neg_root_mean_squared_error")
                        names = macro_feature_names(bins)
                        store.append(seed, {
                            "key": ikey, "record_type": "importance", "seed": seed,
                            "budget": budget, "capacity": cap, "bins": bins, "edge_assignment": edge,
                            "feature_names": names,
                            "importances_mean": [round(float(v), 6) for v in result.importances_mean],
                            "importances_std": [round(float(v), 6) for v in result.importances_std],
                        })
                        existing[ikey] = True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/experiment2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--bins", type=int, nargs="+", default=list(DEFAULT_BINS))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--capacity", default="medium", choices=CAPACITY_ORDER,
                        help="Capacity preset for the main sweep (every budget).")
    parser.add_argument("--heatmap-capacities", nargs="*", default=["small", "large"], choices=CAPACITY_ORDER)
    parser.add_argument("--heatmap-budgets", type=int, nargs="*", default=[250, 5000])
    parser.add_argument("--edge-assignments", nargs="+", default=list(EDGE_ASSIGNMENTS), choices=EDGE_ASSIGNMENTS)
    parser.add_argument("--local-budgets", type=int, nargs="*", default=list(DEFAULT_BUDGETS),
                        help="Budgets at which the local control is run for the macro feature set.")
    parser.add_argument("--importance-bins", type=int, nargs="*", default=[8])
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--quick", action="store_true",
                        help="Two-seed demo over 1/8/64 bins at two budgets (about a minute). Overrides the "
                             "grid options and writes to results/quick/experiment2 unless --output is given.")
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.bins, args.budgets = [1000, 2000], [1, 8, 64], [250, 2000]
        args.heatmap_capacities, args.heatmap_budgets, args.local_budgets = ["small"], [2000], [2000]
        args.n_val, args.n_test = 500, 1000
        if args.output == Path("results/experiment2"):
            args.output = Path("results/quick/experiment2")

    validate_seed_spacing(args.seeds)
    for bins in args.bins:
        validate_bins(bins, "left")   # 1..64; the interior series skips bins > 32 itself.
    if sorted(args.budgets) != args.budgets or len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("Budgets must be strictly increasing")

    store = ResultStore(args.output, "experiment2")
    config = {
        "seeds": args.seeds, "bins": args.bins, "budgets": args.budgets, "train_pool": max(args.budgets),
        "capacity": args.capacity, "heatmap_capacities": args.heatmap_capacities,
        "heatmap_budgets": args.heatmap_budgets, "edge_assignments": args.edge_assignments,
        "local_budgets": args.local_budgets, "importance_bins": args.importance_bins,
        "n_val": args.n_val, "n_test": args.n_test, "stream_offsets": STREAM_OFFSETS,
    }
    store.check_consistent(config, ("train_pool", "n_val", "n_test"))
    store.write_metadata(config)
    started = time.perf_counter()
    for seed in args.seeds:
        run_seed(seed, store, bins_list=tuple(args.bins), budgets=tuple(args.budgets),
                 capacity=args.capacity, heatmap_capacities=tuple(args.heatmap_capacities),
                 heatmap_budgets=tuple(args.heatmap_budgets), edge_assignments=tuple(args.edge_assignments),
                 n_val=args.n_val, n_test=args.n_test, importance_bins=tuple(args.importance_bins),
                 local_budgets=tuple(args.local_budgets))
    print(f"Experiment 2 finished in {time.perf_counter() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
