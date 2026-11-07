"""Experiment 1: robustness across seeds, data budgets and model capacities.

For every seed we simulate a training pool, a validation split, an
in-distribution test split and a shifted-fault test split (see
:mod:`macro_prediction.data` for the random streams). For every training
budget (a nested prefix of the pool), capacity preset and representation we
fit one histogram gradient-boosting regressor on exactly the same episodes
and evaluate it on exactly the same validation/test episodes. The random
projection is re-fitted on each budget's training subset only.

Two tasks are evaluated: the *global* remaining-displacement target, and the
*local* centre-value-after-one-sweep control, which should favour the raw
state. Costs are measured as wall-clock seconds for feature extraction,
model fitting and test-set inference; the realised model size (trees,
leaves, nodes) is recorded as a parameter-count proxy.

The script is resumable: every finished (seed, task, budget, capacity,
representation) record is appended to ``raw/experiment1-seed-<seed>.jsonl``
and skipped on re-runs.

Example::

    uv run macro-experiment1 --output results/experiment1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_SEEDS, SeedData, STREAM_OFFSETS, generate_seed_data, validate_seed_spacing
from .features import BINS, GLOBAL_DIM, REGIONAL_PER_BIN, RandomProjection, macro_features, raw_features
from .models import CAPACITY_ORDER, EvalSet, fit_and_evaluate
from .store import ResultStore

REPRESENTATIONS = ("Full state", "Macroscopic", "Random projection", "Initial disorder", "Full + macro")
DEFAULT_BUDGETS = (250, 750, 2000, 5000)
DEFAULT_CAPACITIES = ("small", "medium", "large", "xlarge")
DEFAULT_RESTRICTIONS = {"xlarge": (5000,)}   # The largest preset is costly; run at 5,000 only.
DEFAULT_LOCAL_CAPACITIES = ("small", "medium", "large")
TIMING_REPEATS = 5


def _timed(fn, *args, repeats: int = TIMING_REPEATS):
    """Return ``(result, median_seconds)``; the median damps timer noise for ms-scale work."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def build_representations(data: SeedData, budget: int, *, bins: int, seed: int) -> tuple[dict, dict]:
    """Return ``{name: {split: features}}`` and per-representation extraction costs.

    Everything that is *fitted* (the projection's scaler and matrix) uses only
    the first ``budget`` training episodes. Extraction cost is the median of
    five repeats for the training subset (``feature_seconds_train``) and for
    the 2,000-episode test split (``feature_seconds_test``).
    """
    splits = {"train": data.train, "val": data.val, "test": data.test, "shift": data.shift}
    rows = {name: (s.arrays[:budget], s.success[:budget]) if name == "train" else (s.arrays, s.success)
            for name, s in splits.items()}

    feats: dict[str, dict[str, np.ndarray]] = {r: {} for r in REPRESENTATIONS}
    cost: dict[str, dict[str, float]] = {r: {} for r in REPRESENTATIONS}

    raw = {}
    for split, (x, p) in rows.items():
        raw[split], t = _timed(raw_features, x, p)
        feats["Full state"][split] = raw[split]
        if split in ("train", "test"):
            cost["Full state"][f"feature_seconds_{split}"] = t

    for split, (x, p) in rows.items():
        macro, t = _timed(lambda x_, p_: macro_features(x_, p_, bins=bins), x, p)
        feats["Macroscopic"][split] = macro
        feats["Initial disorder"][split] = macro[:, :1].astype(np.float64)
        feats["Full + macro"][split] = np.hstack((raw[split], macro))
        if split in ("train", "test"):
            cost["Macroscopic"][f"feature_seconds_{split}"] = t
            cost["Initial disorder"][f"feature_seconds_{split}"] = t
            cost["Full + macro"][f"feature_seconds_{split}"] = t + cost["Full state"][f"feature_seconds_{split}"]

    n_components = GLOBAL_DIM + REGIONAL_PER_BIN * bins
    projection, t_fit = _timed(
        lambda: RandomProjection(n_components, seed + STREAM_OFFSETS["project"]).fit(raw["train"]))
    for split in rows:
        feats["Random projection"][split], t = _timed(projection.transform, raw[split])
        if split == "train":
            cost["Random projection"]["feature_seconds_train"] = t_fit + t + cost["Full state"]["feature_seconds_train"]
        elif split == "test":
            cost["Random projection"]["feature_seconds_test"] = t + cost["Full state"]["feature_seconds_test"]
    return feats, cost


def run_seed(
    seed: int,
    store: ResultStore,
    *,
    budgets: tuple[int, ...],
    capacities: tuple[str, ...],
    restrictions: dict[str, tuple[int, ...]],
    local_capacities: tuple[str, ...],
    bins: int,
    n_val: int,
    n_test: int,
    noise_realizations: int,
    with_local: bool,
) -> None:
    existing = store.load(seed)
    data = generate_seed_data(seed, n_train=max(budgets), n_val=n_val, n_test=n_test,
                              n_floor_realizations=noise_realizations, with_local=with_local)

    for task, floor in data.noise_floor.items():
        key = f"noise_floor|{task}"
        if key not in existing:
            store.append(seed, {"key": key, "record_type": "noise_floor", "seed": seed,
                                "task": task, **floor})

    for budget in budgets:
        feats, cost = build_representations(data, budget, bins=bins, seed=seed)
        for capacity in capacities:
            allowed = restrictions.get(capacity)
            if allowed is not None and budget not in allowed:
                continue
            for task in ("global", "local") if with_local else ("global",):
                if task == "local" and capacity not in local_capacities:
                    continue
                for name in REPRESENTATIONS:
                    key = f"{task}|{budget}|{capacity}|{name}"
                    if key in existing:
                        continue
                    if task == "global":
                        y_train = data.train.target[:budget]
                        eval_sets = [EvalSet("val", feats[name]["val"], data.val.target),
                                     EvalSet("test", feats[name]["test"], data.test.target),
                                     EvalSet("shift", feats[name]["shift"], data.shift.target)]
                    else:
                        y_train = data.train.local_target[:budget]
                        eval_sets = [EvalSet("val", feats[name]["val"], data.val.local_target),
                                     EvalSet("test", feats[name]["test"], data.test.local_target)]
                    record = fit_and_evaluate(capacity, seed, feats[name]["train"], y_train, eval_sets)
                    record.pop("_model")
                    record.update({
                        "key": key, "record_type": "fit", "seed": seed, "task": task,
                        "budget": budget, "representation": name, "bins": bins, **cost[name],
                    })
                    store.append(seed, record)
                    existing[key] = record
                    print(f"seed={seed} {task:6s} n={budget:5d} {capacity:6s} {name:18s} "
                          f"val={record['val_rmse']:.3f} test={record['test_rmse']:.3f} "
                          f"fit={record['fit_seconds']:.2f}s", flush=True)


def parse_restrictions(items: list[str]) -> dict[str, tuple[int, ...]]:
    """Parse ``capacity=budget[,budget...]`` items into a dict."""
    out = {}
    for item in items:
        capacity, _, budgets = item.partition("=")
        if capacity not in CAPACITY_ORDER or not budgets:
            raise argparse.ArgumentTypeError(f"Bad restriction {item!r}; use e.g. xlarge=5000")
        out[capacity] = tuple(int(b) for b in budgets.split(","))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/experiment1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--capacities", nargs="+", default=list(DEFAULT_CAPACITIES), choices=CAPACITY_ORDER)
    parser.add_argument("--restrict", nargs="*", default=["xlarge=5000"],
                        help="Limit a capacity to given budgets, e.g. xlarge=5000. Pass no value to lift all limits.")
    parser.add_argument("--local-capacities", nargs="*", default=list(DEFAULT_LOCAL_CAPACITIES), choices=CAPACITY_ORDER,
                        help="Capacities evaluated on the local control task.")
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--noise-realizations", type=int, default=32,
                        help="Re-simulations per test episode for the irreducible-error estimate (0 to skip).")
    parser.add_argument("--no-local", action="store_true", help="Skip the local control task.")
    parser.add_argument("--quick", action="store_true",
                        help="Two-seed, two-budget, small/medium demo (about a minute). Overrides the grid "
                             "options and writes to results/quick/experiment1 unless --output is given.")
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.budgets = [1000, 2000], [250, 2000]
        args.capacities, args.local_capacities, args.restrict = ["small", "medium"], ["small", "medium"], []
        args.n_val, args.n_test, args.noise_realizations = 500, 1000, 8
        if args.output == Path("results/experiment1"):
            args.output = Path("results/quick/experiment1")

    validate_seed_spacing(args.seeds)
    if sorted(args.budgets) != args.budgets or len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("Budgets must be strictly increasing")
    restrictions = parse_restrictions(args.restrict or [])

    store = ResultStore(args.output, "experiment1")
    config = {
        "seeds": args.seeds, "budgets": args.budgets, "train_pool": max(args.budgets),
        "capacities": args.capacities, "restrictions": restrictions,
        "local_capacities": args.local_capacities, "bins": args.bins, "n_val": args.n_val,
        "n_test": args.n_test, "noise_realizations": args.noise_realizations,
        "with_local": not args.no_local, "stream_offsets": STREAM_OFFSETS,
    }
    store.check_consistent(config, ("train_pool", "n_val", "n_test", "bins"))
    store.write_metadata(config)
    started = time.perf_counter()
    for seed in args.seeds:
        run_seed(seed, store, budgets=tuple(args.budgets), capacities=tuple(args.capacities),
                 restrictions=restrictions, local_capacities=tuple(args.local_capacities),
                 bins=args.bins, n_val=args.n_val, n_test=args.n_test,
                 noise_realizations=args.noise_realizations, with_local=not args.no_local)
    print(f"Experiment 1 finished in {time.perf_counter() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
