"""Experiment 2b: emergent predictive structure (approximate closure test).

Question
--------
Do microscopic configurations with (nearly) the same macroscopic
representation have the same macroscopic future? Formally, for a candidate
abstraction ``Z = phi(X)`` and horizon ``tau``, how close is
``P(Z_{t+tau} | X_t)`` to ``P(Z_{t+tau} | Z_t)``? If the sorting dynamics
approximately close over ``phi``, then the microscopic details left out of
``Z_t`` add little information about the future macrostate; if ``phi`` is
merely a convenient set of coordinates for a learner, they add a lot. No
learner is involved in this test.

Estimator
---------
For each configuration ``i`` we simulate ``R`` independent futures and record
the macroscopic future at every sweep ``tau = 1..H``. Let ``mu_i(tau)`` be the
realisation mean and ``s2_i(tau)`` the realisation variance. Then

* ``V_irr(tau) = mean_i s2_i``                      irreducible (dice) variance,
* ``V_total(tau) = Var_i(mu_i) - V_irr / R``       variance of the true
  conditional means ``E[future | X_i]`` across configurations (unbiased),
* for an abstraction ``phi`` with neighbour sets ``N_k(i)`` (the ``k`` nearest
  *distinct* configurations in standardized ``phi``-space),
  ``excess_phi(tau) = mean_{i, j in N_k(i)} (mu_i - mu_j)^2 / 2 - V_irr / R``,
  which estimates ``E_Z[ Var( E[future | X] | Z ) ]``: the variance in the
  true conditional mean that remains once the macrostate is fixed.
* ``closure_phi(tau) = 1 - excess_phi / V_total``: 1 means the macrostate
  determines the future as well as the microstate does; 0 means it carries no
  information about which configurations share a future.

Neighbour matching replaces exact "same macrostate" cells, which do not exist
for continuous features. It is biased against high-dimensional ``phi``
(neighbours are farther apart), so every record also carries the mean
neighbour distance relative to the mean pairwise distance, and the fair
comparisons are between representations of matched size (global vs. random-3,
macro-8 vs. random-27, ...). Two refinements test whether more detail helps:
``+raw`` re-ranks the ``k0`` nearest macro neighbours by raw-state distance
(does microscopic detail beyond the macrostate matter?), and ``+history``
re-ranks them by the previous sweep's global summaries (is the coarse-grained
dynamics non-Markov in the instantaneous macrostate?).

The "present" configuration is the state after ``present_sweeps`` sweeps of
an episode drawn from the usual episode distribution, so that a one-sweep
history exists and the configurations are mid-recovery rather than freshly
scrambled. The future is measured by the three global summaries at each sweep
(``global3``) and by the global target (``target``, mean displacement), which
is a rescaling of the first summary.

Example::

    uv run macro-experiment2b --output results/experiment2b
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data import DEFAULT_SEEDS, STREAM_OFFSETS, validate_seed_spacing
from .features import (GLOBAL_DIM, N, REGIONAL_PER_BIN, RandomProjection, global_features,
                       macro_features, raw_features)
from .simulator import HORIZON, _sweep, make_episodes
from .store import ResultStore

DEFAULT_BINS = (1, 2, 4, 8, 16, 32)
DEFAULT_RANDOM_DIMS = (3, 6, 27, 99)
MEASURES = ("target", "global3")


def simulate_futures(arrays: np.ndarray, success: np.ndarray, rng: np.random.Generator,
                     *, horizon: int, n_realizations: int) -> np.ndarray:
    """Global summaries after each of ``horizon`` sweeps, for ``n_realizations`` futures.

    Returns an array of shape ``(R, horizon, n_configs, 3)``.
    """
    out = np.empty((n_realizations, horizon, len(arrays), GLOBAL_DIM), dtype=np.float64)
    for r in range(n_realizations):
        a = arrays.copy()
        for t in range(horizon):
            _sweep(a, success, rng)
            out[r, t] = global_features(a, success)
    return out


def present_configurations(seed: int, n_configs: int, present_sweeps: int):
    """Episodes advanced by ``present_sweeps`` sweeps, plus the previous sweep's global summaries."""
    rng = np.random.default_rng(seed + STREAM_OFFSETS["closure_config"])
    arrays, success = make_episodes(n_configs, rng)
    history = global_features(arrays, success)
    for t in range(present_sweeps):
        history = global_features(arrays, success)
        _sweep(arrays, success, rng)
    return arrays, success, history


def candidate_representations(arrays, success, seed: int, bins_list, random_dims) -> dict[str, np.ndarray]:
    raw = raw_features(arrays, success)
    reps = {"initial_disorder": global_features(arrays, success)[:, :1],
            "global": global_features(arrays, success)}
    for b in bins_list:
        reps[f"macro-{b}"] = macro_features(arrays, success, bins=b)
    reps["regional-8"] = macro_features(arrays, success, bins=8, include_global=False)
    for d in random_dims:
        reps[f"random-{d}"] = RandomProjection(d, seed + STREAM_OFFSETS["closure_project"]).fit(raw).transform(raw)
    reps["raw"] = raw
    return reps


def _standardize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    sd = z.std(axis=0)
    sd[sd == 0] = 1.0
    return (z - z.mean(axis=0)) / sd


def nearest_distinct(z: np.ndarray, arrays: np.ndarray, k: int, *, refine_by=None,
                     k0: int | None = None, rng: np.random.Generator | None = None):
    """Indices of the ``k`` nearest distinct configurations in standardized ``z``-space.

    With ``refine_by`` an array, the ``k0`` nearest in ``z`` are re-ranked by
    distance in the refinement space and the ``k`` nearest of those are kept.
    With ``refine_by="random"``, ``k`` of the ``k0`` are chosen at random:
    this is the control for a refinement, since it inflates the ``z``-distance
    of the kept neighbours in the same way without using any information.
    Returns ``(neighbours, distances, relative_distance)``: the neighbour
    indices, their distances in ``z``-space, and the mean neighbour distance
    divided by the mean pairwise distance in ``z``-space.
    """
    zs = _standardize(z)
    n_query = k0 if refine_by is not None else k
    random_pick = isinstance(refine_by, str)
    rng = rng or np.random.default_rng(0)
    # Query with slack so that enough candidates survive the distinctness filter
    # (identical microstates, e.g. two already-sorted arrays with the same
    # fault profile, are excluded because they are the same configuration).
    nn = NearestNeighbors(n_neighbors=min(3 * n_query + 10, len(zs))).fit(zs)
    dist, idx = nn.kneighbors(zs)
    neighbours = np.empty((len(zs), k), dtype=np.int64)
    kept_dist = np.empty((len(zs), k), dtype=np.float64)
    ref = _standardize(refine_by) if (refine_by is not None and not random_pick) else None
    for i in range(len(zs)):
        cand = [(d, j) for d, j in zip(dist[i], idx[i]) if j != i and not np.array_equal(arrays[j], arrays[i])]
        if not cand:  # Degenerate tiny inputs only: fall back to any other configuration.
            cand = [(d, j) for d, j in zip(dist[i], idx[i]) if j != i]
        cand = cand[:n_query]
        if ref is not None:
            cand = sorted(cand, key=lambda dj: np.sum((ref[dj[1]] - ref[i]) ** 2))
        elif random_pick and len(cand) > k:
            cand = [cand[j] for j in rng.choice(len(cand), size=k, replace=False)]
        cand = cand[:k]
        if len(cand) < k:
            cand = cand + [cand[-1]] * (k - len(cand))
        neighbours[i] = [j for _, j in cand]
        kept_dist[i] = [d for d, _ in cand]
    sample = np.random.default_rng(0).choice(len(zs), size=min(2000, len(zs)), replace=False)
    pair = np.sqrt(((zs[sample, None, :] - zs[None, sample, :]) ** 2).sum(-1))
    mean_pair = pair[np.triu_indices(len(sample), 1)].mean()
    relative = float(kept_dist.mean() / mean_pair) if mean_pair > 0 else float("nan")
    return neighbours, kept_dist, relative


def closure_metrics(mu: np.ndarray, s2: np.ndarray, neighbours: np.ndarray, distances: np.ndarray,
                    n_realizations: int) -> dict:
    """Excess conditional-mean variance given the matched macrostate; see module docstring.

    ``mu`` and ``s2`` have shape ``(n_configs, d)``; the ``d`` components are
    summed after the caller has scaled them to comparable units.

    ``excess`` is the mean over matched pairs of the squared difference of
    realisation means, halved and corrected for realisation noise. It is
    biased upward for high-dimensional representations, whose nearest
    neighbours are farther apart; the records therefore carry the neighbour
    distance, and the experiment also reports the estimate at several sample
    sizes so that convergence can be judged. (A distance-extrapolated
    estimator was tried and rejected: in high dimensions neighbour distances
    concentrate, and the extrapolation to zero distance is unstable.)
    """
    v_irr = s2.mean(axis=0)
    v_total = mu.var(axis=0, ddof=1) - v_irr / n_realizations
    noise = (v_irr / n_realizations).sum()
    diff2 = ((mu[:, None, :] - mu[neighbours]) ** 2).sum(axis=2)   # (n, k), summed over components
    excess = float((diff2 / 2 - noise).mean())
    v_total_sum = float(v_total.sum())
    return {
        "v_irr": float(v_irr.sum()),
        "v_total": v_total_sum,
        "excess": excess,
        "closure_index": float(1 - excess / v_total_sum) if v_total_sum > 0 else float("nan"),
        "mean_neighbour_distance": float(distances.mean()),
    }


def run_seed(seed: int, store: ResultStore, *, n_configs, n_realizations, horizon, k, k0,
             present_sweeps, bins_list, random_dims) -> None:
    existing = store.load(seed)
    subsample_sizes = sorted({n_configs // 4, n_configs // 2, n_configs})
    if existing and all(f"closure|{name}|{m}|{tau}|{n_configs}" in existing
                        for name in ["raw", "global", "macro-8+random"]
                        for m in MEASURES for tau in range(1, horizon + 1)):
        print(f"seed={seed} already complete", flush=True)
        return

    started = time.perf_counter()
    arrays, success, history = present_configurations(seed, n_configs, present_sweeps)
    futures = simulate_futures(arrays, success, np.random.default_rng(seed + STREAM_OFFSETS["closure_future"]),
                               horizon=horizon, n_realizations=n_realizations)
    reps = candidate_representations(arrays, success, seed, bins_list, random_dims)
    print(f"seed={seed} simulated {n_configs} configs x {n_realizations} futures in "
          f"{time.perf_counter() - started:.1f}s", flush=True)

    # Per-measure realisation means and variances, scaled to comparable units.
    mu_all = futures.mean(axis=0)                      # (H, n, 3)
    s2_all = futures.var(axis=0, ddof=1)               # (H, n, 3)
    measures = {}
    for tau in range(horizon):
        target_mu, target_s2 = mu_all[tau, :, :1] * N, s2_all[tau, :, :1] * N ** 2
        scale = mu_all[tau].std(axis=0)
        scale[scale == 0] = 1.0
        measures[("target", tau + 1)] = (target_mu, target_s2)
        measures[("global3", tau + 1)] = (mu_all[tau] / scale, s2_all[tau] / scale ** 2)

    # Plain matchings at the full sample and at smaller subsamples (convergence
    # check), plus refinements with their random-pick controls.
    matchings: list[tuple[str, np.ndarray, object, int]] = []
    for name, z in reps.items():
        for n_used in subsample_sizes:
            matchings.append((name, z, None, n_used))
    for base in ("global", "macro-8"):
        matchings.append((f"{base}+raw", reps[base], reps["raw"], n_configs))
        matchings.append((f"{base}+history", reps[base], history, n_configs))
        matchings.append((f"{base}+random", reps[base], "random", n_configs))
    # Control for "+history": re-ranking by the *present* global summaries. If
    # it matches "+history", the gain is tighter matching of the present
    # macrostate, not memory of the past.
    matchings.append(("macro-8+global", reps["macro-8"], reps["global"], n_configs))

    for name, z, refine, n_used in matchings:
        keys = [f"closure|{name}|{m}|{tau}|{n_used}" for m in MEASURES for tau in range(1, horizon + 1)]
        if all(key in existing for key in keys):
            continue
        rows = slice(0, n_used)
        # Distinctness is judged on the full microstate (values and fault profile).
        neighbours, distances, rel_dist = nearest_distinct(
            z[rows], reps["raw"][rows], k, refine_by=None if refine is None else (refine if isinstance(refine, str) else refine[rows]),
            k0=k0 if refine is not None else None, rng=np.random.default_rng(seed + 1))
        for (measure, tau), (mu, s2) in measures.items():
            key = f"closure|{name}|{measure}|{tau}|{n_used}"
            if key in existing:
                continue
            metrics = closure_metrics(mu[rows], s2[rows], neighbours, distances, n_realizations)
            record = {"key": key, "record_type": "closure", "seed": seed, "representation": name,
                      "base": name.split("+")[0], "n_features": int(z.shape[1]),
                      "refined_by": "none" if refine is None else name.split("+")[1],
                      "measure": measure, "tau": tau, "k": k, "n_used": n_used, "n_configs": n_configs,
                      "n_realizations": n_realizations, "relative_neighbour_distance": round(rel_dist, 5),
                      **{kk: round(v, 8) for kk, v in metrics.items()}}
            store.append(seed, record)
            existing[key] = record
        last = existing[f"closure|{name}|target|{horizon}|{n_used}"]
        print(f"seed={seed} {name:16s} dim={z.shape[1]:3d} n={n_used:5d} rel.dist={rel_dist:.3f} "
              f"closure(target, tau={horizon})={last['closure_index']:.3f}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/experiment2b"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-configs", type=int, default=10000)
    parser.add_argument("--realizations", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--k", type=int, default=10, help="Matched neighbours per configuration.")
    parser.add_argument("--k0", type=int, default=50, help="Candidate neighbours before refinement.")
    parser.add_argument("--present-sweeps", type=int, default=2)
    parser.add_argument("--bins", type=int, nargs="+", default=list(DEFAULT_BINS))
    parser.add_argument("--random-dims", type=int, nargs="+", default=list(DEFAULT_RANDOM_DIMS))
    parser.add_argument("--quick", action="store_true",
                        help="Two seeds, 2,000 configurations, 8 futures (under a minute).")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.n_configs, args.realizations = [1000, 2000], 2000, 8
        if args.output == Path("results/experiment2b"):
            args.output = Path("results/quick/experiment2b")
    validate_seed_spacing(args.seeds)
    if 8 not in args.bins:
        raise SystemExit("--bins must include 8 (used for the refinement tests)")

    store = ResultStore(args.output, "experiment2b")
    config = {"seeds": args.seeds, "n_configs": args.n_configs, "realizations": args.realizations,
              "horizon": args.horizon, "k": args.k, "k0": args.k0, "present_sweeps": args.present_sweeps,
              "bins": args.bins, "random_dims": args.random_dims, "stream_offsets": STREAM_OFFSETS}
    store.check_consistent(config, ("n_configs", "realizations", "horizon", "k", "k0", "present_sweeps"))
    store.write_metadata(config)
    started = time.perf_counter()
    for seed in args.seeds:
        run_seed(seed, store, n_configs=args.n_configs, n_realizations=args.realizations, horizon=args.horizon,
                 k=args.k, k0=args.k0, present_sweeps=args.present_sweeps, bins_list=tuple(args.bins),
                 random_dims=tuple(args.random_dims))
    print(f"Experiment 2b finished in {time.perf_counter() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
