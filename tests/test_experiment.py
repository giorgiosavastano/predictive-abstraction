"""Lightweight tests for the scientific assumptions and the CLI output."""

import csv
import json

import numpy as np
import pytest

from macro_prediction.experiment import (
    N,
    make_episodes,
    one_sweep_central_value,
    representations,
    run_experiment,
    simulate,
)


def test_episodes_are_permutations_and_reliabilities_valid():
    """Every episode must retain exactly the original array values."""
    x, p = make_episodes(12, np.random.default_rng(7))
    assert x.shape == p.shape == (12, N)
    assert np.array_equal(np.sort(x, axis=1), np.tile(np.arange(N), (12, 1)))
    assert np.all((p >= 0) & (p <= 1))


def test_sorted_array_stays_sorted():
    """A sorted array should be an absorbing state under local sorting."""
    x = np.tile(np.arange(N), (5, 1))
    p = np.ones((5, N), dtype=np.float32)
    assert np.all(simulate(x, p, np.random.default_rng(0)) == 0)


def test_representations_shapes_and_bin_count():
    """Macro dimensionality should scale linearly with the bin count."""
    x, p = make_episodes(5, np.random.default_rng(42))
    for bins in (1, 4, 8, 16):
        raw, macro, scalar = representations(x, p, bins=bins)
        assert raw.shape == (5, 128)
        assert macro.shape == (5, 3 + 3 * bins)
        assert scalar.shape == (5, 1)
        assert np.isfinite(macro).all()


def test_macro_summary_is_input_only():
    """Features cannot change because a different future trajectory was sampled."""
    x, p = make_episodes(8, np.random.default_rng(31))
    before = representations(x, p)[1].copy()
    simulate(x, p, np.random.default_rng(40))
    one_sweep_central_value(x, p, np.random.default_rng(41))
    after = representations(x, p)[1]
    assert np.array_equal(before, after)


def test_invalid_bins():
    x, p = make_episodes(3, np.random.default_rng(5))
    with pytest.raises(ValueError):
        representations(x, p, bins=0)


def test_small_end_to_end_run_creates_outputs(tmp_path):
    """Smoke test: data generation, model fitting, and exported artifacts."""
    records, local = run_experiment(output=tmp_path, n_test=12,
                                    train_sizes=(25,), seed=5)
    assert len(records) == 5  # Five representations of the global target.
    assert len(local) == 3  # Three representations of the local target.
    for name in ("learning_curves.csv", "local_control.csv",
                 "learning_curves.png", "run.json"):
        assert (tmp_path / name).is_file()
    with (tmp_path / "learning_curves.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 5
    meta = json.loads((tmp_path / "run.json").read_text())
    assert meta["seed"] == 5
    assert meta["bins"] == 8
