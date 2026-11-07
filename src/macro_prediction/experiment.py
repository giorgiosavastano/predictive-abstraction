"""Run the baseline experiment on task-aware coarse-graining.

This is a simplified stochastic local-sorting model, NOT a reproduction of
Zhang, Goldstein and Levin's self-sorting-array experiments. Its initial
state and exchange-reliability profile are known to every representation.

The simulator and feature code now live in :mod:`macro_prediction.simulator`
and :mod:`macro_prediction.features`; they are re-exported here so that the
original import paths and the exact baseline behaviour are preserved. The
multi-seed experiments live in :mod:`macro_prediction.experiment1` and
:mod:`macro_prediction.experiment2`.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring an interactive desktop session.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .features import BINS, representations
from .models import score
from .simulator import HORIZON, N, make_episodes, one_sweep_central_value, simulate

__all__ = [
    "BINS", "HORIZON", "N", "SEED", "TRAIN_SIZES", "N_TEST", "DEFAULT_OUTPUT",
    "make_episodes", "simulate", "representations", "one_sweep_central_value",
    "score", "run_experiment", "main",
]

SEED = 20261106
TRAIN_SIZES = (250, 750, 2000, 5000)
N_TEST = 2000
DEFAULT_OUTPUT = Path("results/runs/latest")


def run_experiment(
    *,
    output: Path = DEFAULT_OUTPUT,
    seed: int = SEED,
    train_sizes: tuple[int, ...] = TRAIN_SIZES,
    n_test: int = N_TEST,
    bins: int = BINS,
) -> tuple[list[dict], list[dict]]:
    """Run all comparisons and save CSVs and a figure in ``output``.

    The baseline defaults preserve the original experiment's RNG ordering.
    Varying the seed and bin count enables simple replication and sensitivity
    analyses while leaving the baseline files unmodified.

    Known limitation kept for backwards compatibility: the random-projection
    scaler is fitted on the whole training pool (``max(train_sizes)``), not on
    each budget's subset. The multi-seed experiments fit it per budget.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if seed < 0 or n_test < 1 or not train_sizes or min(train_sizes) < 1:
        raise ValueError("Provide non-negative seed and positive sample counts")
    if len(set(train_sizes)) != len(train_sizes):
        raise ValueError("Training sizes must be unique")

    # Separate random streams prevent changes to the training simulator from
    # silently altering the test episodes, and make the run reproducible.
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)
    rng_shift = np.random.default_rng(seed + 2)

    train_x, train_p = make_episodes(max(train_sizes), rng_train)
    test_x, test_p = make_episodes(n_test, rng_test)
    shift_x, shift_p = make_episodes(n_test, rng_shift, shifted=True)

    y_train = simulate(train_x, train_p, rng_train)
    y_test = simulate(test_x, test_p, rng_test)
    y_shift = simulate(shift_x, shift_p, rng_shift)

    train_full, train_macro, train_scalar = representations(train_x, train_p, bins=bins)
    test_full, test_macro, test_scalar = representations(test_x, test_p, bins=bins)
    shift_full, shift_macro, shift_scalar = representations(shift_x, shift_p, bins=bins)

    # Random projection is a same-size compressed representation that has not
    # been designed using knowledge of sorting. It is fixed *before* training.
    # Standardize the raw coordinates using the training sample only.
    # This avoids accidentally giving initial locations greater weight.
    scale_full = StandardScaler().fit(train_full)
    rng_project = np.random.default_rng(seed + 3)
    matrix = rng_project.normal(size=(128, 3 + 3 * bins)).astype(np.float32) / np.sqrt(3 + 3 * bins)
    train_random = scale_full.transform(train_full) @ matrix
    test_random = scale_full.transform(test_full) @ matrix
    shift_random = scale_full.transform(shift_full) @ matrix

    # Additional diagnostic: if full-state + macro features wins substantially,
    # the macroscopic summary discarded useful predictive information.
    modes = {
        "Full state": (train_full, test_full, shift_full),
        "Macroscopic": (train_macro, test_macro, shift_macro),
        "Random projection": (train_random, test_random, shift_random),
        "Initial disorder": (train_scalar, test_scalar, shift_scalar),
        "Full + macro": (
            np.hstack((train_full, train_macro)),
            np.hstack((test_full, test_macro)),
            np.hstack((shift_full, shift_macro)),
        ),
    }

    records = []
    for n_train in train_sizes:
        for name, (features_train, features_test, features_shift) in modes.items():
            start = time.perf_counter()
            # All models have the same trees, leaves, depth and stopping rules.
            # Unlike a model with a separate feature extractor, this learner
            # makes it easy to compare the predictive value of representations.
            model = HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=9,
                min_samples_leaf=12,
                learning_rate=0.075,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=seed,
            )
            model.fit(features_train[:n_train], y_train[:n_train])
            fit_seconds = time.perf_counter() - start
            pred_test = model.predict(features_test)
            pred_shift = model.predict(features_shift)
            rmse, r2 = score(y_test, pred_test)
            shift_rmse, shift_r2 = score(y_shift, pred_shift)
            records.append({
                "train_examples": n_train, "representation": name,
                "test_rmse": round(rmse, 5), "test_r2": round(r2, 5),
                "shift_rmse": round(shift_rmse, 5), "shift_r2": round(shift_r2, 5),
                "fit_seconds": round(fit_seconds, 3),
            })
            print(f"N={n_train:5d} {name:18s} test RMSE={rmse:.3f} "
                  f"shift RMSE={shift_rmse:.3f} fit={fit_seconds:.2f}s", flush=True)

    csv_path = output / "learning_curves.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # Falsification control: for a future *local* value, a macroscopic summary
    # ought to lose information. This shows that no representation is best for
    # every task, rather than presenting a one-sided comparison.
    local_train = one_sweep_central_value(
        train_x, train_p, np.random.default_rng(seed + 10)
    )
    local_test = one_sweep_central_value(
        test_x, test_p, np.random.default_rng(seed + 11)
    )
    local_records = []
    for name in ("Full state", "Macroscopic", "Initial disorder"):
        training_rep, testing_rep, _ = modes[name]
        local_model = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=9, min_samples_leaf=12,
            learning_rate=0.075, l2_regularization=2.0,
            early_stopping=False, random_state=seed,
        )
        local_model.fit(training_rep, local_train)
        rmse, r2 = score(local_test, local_model.predict(testing_rep))
        local_records.append({"representation": name, "test_rmse": rmse, "test_r2": r2})
        print(f"Local task: {name:18s} test RMSE={rmse:.3f}", flush=True)
    with (output / "local_control.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(local_records[0]))
        writer.writeheader()
        writer.writerows(local_records)

    # Report the actual empirical curves, not an idealized theoretical sketch.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name in modes:
        rows = [r for r in records if r["representation"] == name]
        ax.plot([r["train_examples"] for r in rows],
                [r["test_rmse"] for r in rows], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(n) for n in train_sizes])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Test RMSE (remaining displacement, positions)")
    ax.set_title("Does task-relevant coarse-graining aid prediction?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    # Report the variability of the target, so that RMSE can be compared with
    # the difficulty of the task rather than being interpreted in isolation.
    print(f"\nTargets: train mean={y_train.mean():.3f}, sd={y_train.std():.3f}; "
          f"test mean={y_test.mean():.3f}, sd={y_test.std():.3f}")
    # Persist key configuration and software versions alongside every run.
    # This makes it possible to audit how a published figure was generated.
    import sklearn
    import matplotlib as mpl
    (output / "run.json").write_text(json.dumps({
        "seed": seed, "array_size": N, "horizon": HORIZON,
        "bins": bins, "train_sizes": list(train_sizes), "test_episodes": n_test,
        "versions": {
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
        },
        "note": "A simplified local sorting simulator, not a paper replication.",
    }, indent=2) + "\n")
    print(f"Saved: {csv_path}")
    return records, local_records


def main() -> None:
    """Command-line interface for reproducible runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory (never overwrites checked-in baseline).")
    parser.add_argument("--seed", type=int, default=SEED, help="PRNG seed.")
    parser.add_argument("--bins", type=int, default=BINS,
                        help="Number of spatial bins in the macro representation.")
    parser.add_argument("--test-episodes", type=int, default=N_TEST,
                        help="Number of independent episodes in each test split.")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(TRAIN_SIZES),
                        help="Learning-curve budgets, e.g. 250 750 2000 5000.")
    args = parser.parse_args()
    run_experiment(output=args.output, seed=args.seed, bins=args.bins,
                   n_test=args.test_episodes, train_sizes=tuple(args.train_sizes))


if __name__ == "__main__":
    main()
