"""Per-seed datasets with documented random streams and clean splits.

Random-number streams
---------------------
Every stream is ``numpy.random.default_rng(seed + offset)``. The offsets for
the training pool, in-distribution test split, shifted test split, random
projection and local-target trajectories are the ones used by the original
baseline, so a run of the new pipeline with ``seed=20261106``, 8 bins and the
``medium`` capacity reproduces the baseline numbers exactly at the 5,000
budget. New streams (validation and noise-floor) use offsets the baseline
never used.

===========  =======  ===================================================
offset       stream   used for
===========  =======  ===================================================
``+0``       train    training-pool episodes *and* their future trajectories
``+1``       test     in-distribution test episodes and trajectories
``+2``       shift    shifted-fault test episodes and trajectories
``+3``       project  Gaussian random-projection matrix
``+4``       val      validation episodes and trajectories (new)
``+10``      local    local-task trajectories for the training pool
``+11``      local    local-task trajectories for the test split
``+12``      local    local-task trajectories for the validation split (new)
``+20``      floor    repeated trajectories for the noise-floor estimate (new)
===========  =======  ===================================================

Because the offsets span ``+0`` to ``+20``, experiment seeds must be at least
100 apart to keep the streams of different seeds disjoint; the CLI enforces
this. Training-budget subsets are *nested* prefixes of the training pool, as
in the baseline, so every representation and every capacity sees exactly the
same episodes at a given budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulator import conditional_target_statistics, make_episodes, one_sweep_central_value, simulate

STREAM_OFFSETS = {
    "train": 0, "test": 1, "shift": 2, "project": 3, "val": 4,
    "local_train": 10, "local_test": 11, "local_val": 12, "floor": 20,
    # Experiment 2b (closure test): configurations, their futures, projections.
    "closure_config": 30, "closure_future": 31, "closure_project": 32,
}
MIN_SEED_SPACING = 100
DEFAULT_SEEDS = tuple(1000 * k for k in range(1, 11))


@dataclass
class Split:
    arrays: np.ndarray
    success: np.ndarray
    target: np.ndarray                 # Global remaining displacement.
    local_target: np.ndarray | None = None  # Centre value after one sweep.


@dataclass
class SeedData:
    seed: int
    train: Split
    val: Split
    test: Split
    shift: Split
    noise_floor: dict = field(default_factory=dict)


def validate_seed_spacing(seeds: list[int]) -> None:
    ordered = sorted(set(seeds))
    if len(ordered) != len(seeds):
        raise ValueError("Seeds must be unique")
    for a, b in zip(ordered, ordered[1:]):
        if b - a < MIN_SEED_SPACING:
            raise ValueError(
                f"Seeds {a} and {b} are closer than {MIN_SEED_SPACING}; their derived "
                "random streams would overlap")


def generate_seed_data(
    seed: int,
    *,
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 2000,
    n_floor_realizations: int = 0,
    with_local: bool = True,
) -> SeedData:
    """Simulate all splits for one seed using the documented streams."""
    rng = {k: np.random.default_rng(seed + v) for k, v in STREAM_OFFSETS.items()}

    def build(name: str, n: int, *, shifted: bool = False, local_key: str | None = None) -> Split:
        arrays, success = make_episodes(n, rng[name], shifted=shifted)
        target = simulate(arrays, success, rng[name])
        local = None
        if with_local and local_key is not None:
            local = one_sweep_central_value(arrays, success, rng[local_key])
        return Split(arrays, success, target, local)

    train = build("train", n_train, local_key="local_train")
    test = build("test", n_test, local_key="local_test")
    shift = build("shift", n_test, shifted=True)
    val = build("val", n_val, local_key="local_val")

    noise_floor: dict = {}
    if n_floor_realizations > 0:
        mean, sd = conditional_target_statistics(
            test.arrays, test.success, rng["floor"], n_realizations=n_floor_realizations)
        noise_floor["global"] = {
            "n_realizations": n_floor_realizations,
            "irreducible_rmse": float(np.sqrt(np.mean(sd ** 2))),
            "target_sd": float(test.target.std()),
            "rmse_of_conditional_mean_vs_realised": float(np.sqrt(np.mean((mean - test.target) ** 2))),
        }
        if with_local:
            mean_l, sd_l = conditional_target_statistics(
                test.arrays, test.success, rng["floor"],
                n_realizations=n_floor_realizations, target="local")
            noise_floor["local"] = {
                "n_realizations": n_floor_realizations,
                "irreducible_rmse": float(np.sqrt(np.mean(sd_l ** 2))),
                "target_sd": float(test.local_target.std()),
                "rmse_of_conditional_mean_vs_realised": float(np.sqrt(np.mean((mean_l - test.local_target) ** 2))),
            }
    return SeedData(seed, train, val, test, shift, noise_floor)
