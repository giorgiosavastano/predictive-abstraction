"""Stochastic local-sorting simulator and prediction targets.

This is a simplified stochastic odd-even compare-and-swap model, NOT a
reproduction of Zhang, Goldstein and Levin's self-sorting-array experiments.
The functions here were moved verbatim from the original ``experiment.py`` so
that the baseline CLI keeps its exact random-number-generation order.
"""
from __future__ import annotations

import numpy as np

N = 64          # Number of interacting elements in each simulated system.
HORIZON = 12    # Predict the state after 12 full even/odd sorting sweeps.


def make_episodes(n_episodes: int, rng: np.random.Generator, *, shifted: bool = False):
    """Generate independently disturbed arrays and local reliability profiles.

    Every episode starts perfectly sorted. We then scramble a contiguous block
    (sometimes two), allowing the initial severity and geometry to vary.
    A slow region introduces a *spatially localized* fault: compare-and-swap
    attempts crossing its edges have a smaller probability of succeeding.

    For the shifted test, faults are longer and more severe than in training.
    This tests robustness beyond an independent, identically distributed split.
    """
    positions = np.arange(N, dtype=np.int16)
    arrays = np.tile(positions, (n_episodes, 1))
    success = np.full((n_episodes, N), 0.90, dtype=np.float32)

    for row in range(n_episodes):
        # Draw one or two independent scrambled regions, without changing the
        # values in the array as a whole (so perfect ordering remains possible).
        n_regions = 1 if rng.random() < 0.75 else 2
        for _ in range(n_regions):
            width = int(rng.integers(5, 33))
            start = int(rng.integers(0, N - width + 1))
            arrays[row, start : start + width] = rng.permutation(
                arrays[row, start : start + width]
            )

        # All regions usually have 90% exchange reliability. In a damaged
        # interval this falls, delaying but not necessarily preventing repair.
        fault_width = int(rng.integers(5, 21 if not shifted else 31))
        fault_start = int(rng.integers(0, N - fault_width))
        low, high = (0.15, 0.65) if not shifted else (0.05, 0.30)
        success[row, fault_start : fault_start + fault_width] = rng.uniform(low, high)

    # The final entry is only padding: an N-element array has N-1 swap edges.
    success[:, -1] = 1.0
    return arrays, success


def _sweep(a: np.ndarray, success: np.ndarray, rng: np.random.Generator) -> None:
    """Apply one full even/odd sweep of stochastic compare-and-swap in place."""
    n_episodes = len(a)
    for parity in (0, 1):
        left = np.arange(parity, N - 1, 2)
        right = left + 1
        x, y = a[:, left].copy(), a[:, right].copy()
        swap = (x > y) & (rng.random((n_episodes, len(left))) < success[:, left])
        a[:, left] = np.where(swap, y, x)
        a[:, right] = np.where(swap, x, y)


def simulate(arrays: np.ndarray, success: np.ndarray, rng: np.random.Generator):
    """Simulate stochastic local interactions and return remaining disorder.

    One full sweep comprises both parity phases. In each phase, *disjoint*
    neighboring pairs compare values and swap when inverted and the edge's
    random reliability trial succeeds. Each step therefore uses only local
    information, but the resulting array-wide ordering is an emergent outcome.
    """
    a = arrays.copy()
    for _ in range(HORIZON):
        _sweep(a, success, rng)

    # Mean absolute displacement is zero only when the array is fully sorted.
    # Its unit is "array positions per element", which is easy to interpret.
    return np.abs(a - np.arange(N)).mean(axis=1).astype(np.float32)


def one_sweep_central_value(arrays: np.ndarray, success: np.ndarray, rng: np.random.Generator):
    """Counterexample task: predict the value at one particular cell one sweep later.

    This depends strongly on microscopic neighborhood identity. It should NOT
    be well represented by coarse spatial averages, demonstrating that useful
    forgetting must be relative to a specific prediction task.
    """
    a = arrays.copy()
    _sweep(a, success, rng)
    return a[:, N // 2].astype(np.float32)


def conditional_target_statistics(
    arrays: np.ndarray,
    success: np.ndarray,
    rng: np.random.Generator,
    *,
    n_realizations: int = 32,
    target: str = "global",
):
    """Estimate the conditional mean and variance of a stochastic target.

    The target is a random variable even when the initial state is fully
    known, because every exchange attempt is a Bernoulli trial. Re-simulating
    each episode ``n_realizations`` times with fresh randomness gives a Monte
    Carlo estimate of ``E[Y | X0]`` and ``Var[Y | X0]``. The root of the mean
    conditional variance is the *irreducible RMSE* that no predictor using
    only the initial snapshot can beat in expectation. It gives all reported
    RMSE values a floor to be compared against.

    Returns ``(conditional_mean, conditional_sd)`` per episode. The variance
    estimate is the unbiased sample variance across realizations.
    """
    simulate_fn = simulate if target == "global" else one_sweep_central_value
    draws = np.stack([simulate_fn(arrays, success, rng) for _ in range(n_realizations)])
    return draws.mean(axis=0), draws.std(axis=0, ddof=1)
