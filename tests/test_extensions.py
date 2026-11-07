"""Tests for the multi-seed experiment extensions.

Covers feature dimensions and the exact equivalence of the refactored
extractor with the original code, the documented edge-assignment rules,
split generation and seed spacing, deterministic execution, resumability
of the result store, the statistics helpers and the aggregation pipeline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from macro_prediction import analysis, experiment1, experiment2
from macro_prediction.data import (STREAM_OFFSETS, generate_seed_data, validate_seed_spacing)
from macro_prediction.features import (N, RandomProjection, macro_feature_names, macro_features, raw_features,
                                       representation_size_metrics, representations, validate_bins)
from macro_prediction.models import CAPACITIES, EvalSet, fit_and_evaluate
from macro_prediction.simulator import conditional_target_statistics, make_episodes, simulate
from macro_prediction.stats import paired_difference, summarize
from macro_prediction.store import ResultStore


def _original_macro(arrays, success, bins):
    """Verbatim copy of the original baseline feature code, used as an oracle."""
    position = np.arange(N)
    displacement = np.abs(arrays.astype(np.float32) - position) / N
    descents = (arrays[:, :-1] > arrays[:, 1:]).astype(np.float32)
    low_reliability = 1.0 - success[:, :-1]
    global_features = np.column_stack((
        displacement.mean(axis=1),
        descents.mean(axis=1),
        (displacement[:, :-1] * low_reliability).mean(axis=1),
    ))
    features = [global_features]
    for inds in np.array_split(np.arange(N), bins):
        edges = inds[:-1]
        features.append(np.column_stack((
            displacement[:, inds].mean(axis=1),
            descents[:, edges].mean(axis=1),
            low_reliability[:, edges].mean(axis=1),
        )))
    macro = np.concatenate(features, axis=1).astype(np.float32)
    return macro, global_features[:, :1].copy()


@pytest.fixture(scope="module")
def episodes():
    return make_episodes(40, np.random.default_rng(99))


def test_refactored_features_match_original_exactly(episodes):
    x, p = episodes
    for bins in (1, 2, 4, 8, 16, 32):
        raw, macro, scalar = representations(x, p, bins=bins)
        ref_macro, ref_scalar = _original_macro(x, p, bins)
        assert np.array_equal(macro, ref_macro)
        assert np.array_equal(scalar, ref_scalar) and scalar.dtype == ref_scalar.dtype
        assert np.array_equal(macro, macro_features(x, p, bins=bins))
        assert np.array_equal(raw, raw_features(x, p))


def test_feature_dimensions_and_names(episodes):
    x, p = episodes
    for bins in (1, 2, 4, 8, 16, 32, 64):
        for include_global, include_regional in ((True, True), (True, False), (False, True)):
            feats = macro_features(x, p, bins=bins, include_global=include_global,
                                   include_regional=include_regional, edge_assignment="left")
            names = macro_feature_names(bins, include_global=include_global, include_regional=include_regional)
            expected = 3 * include_global + 3 * bins * include_regional
            assert feats.shape == (40, expected)
            assert len(names) == expected
            assert np.isfinite(feats).all()


def test_edge_rules_and_invalid_bins(episodes):
    x, p = episodes
    with pytest.raises(ValueError):
        macro_features(x, p, bins=64, edge_assignment="interior")
    with pytest.raises(ValueError):
        validate_bins(0)
    with pytest.raises(ValueError):
        validate_bins(65, "left")
    with pytest.raises(ValueError):
        macro_features(x, p, bins=8, include_global=False, include_regional=False)
    # With one bin both rules see every one of the 63 edges and agree exactly.
    assert np.array_equal(macro_features(x, p, bins=1, edge_assignment="interior"),
                          macro_features(x, p, bins=1, edge_assignment="left"))
    # Under the left rule every edge is counted exactly once: the
    # edge-count-weighted mean of regional descents equals the global descent fraction.
    for bins in (4, 8, 64):
        feats = macro_features(x, p, bins=bins, edge_assignment="left")
        counts = np.array([len(inds[inds < N - 1]) for inds in np.array_split(np.arange(N), bins)])
        regional_descents = feats[:, 3 + 1::3]
        weighted = (regional_descents * counts).sum(axis=1) / counts.sum()
        assert np.allclose(weighted, feats[:, 1], atol=1e-6)
    # At 64 bins the regional displacement is exactly the per-element displacement.
    feats64 = macro_features(x, p, bins=64, include_global=False, edge_assignment="left")
    assert np.allclose(feats64[:, 0::3], np.abs(x - np.arange(N)) / N, atol=1e-6)


def test_random_projection_fits_on_training_rows_only(episodes):
    x, p = episodes
    raw = raw_features(x, p)
    proj_a = RandomProjection(9, seed=5).fit(raw[:20])
    proj_b = RandomProjection(9, seed=5).fit(raw[:20])
    proj_c = RandomProjection(9, seed=5).fit(raw[:30])
    assert np.array_equal(proj_a.transform(raw), proj_b.transform(raw))
    assert np.array_equal(proj_a.matrix, proj_c.matrix)          # Matrix depends on the seed only...
    assert not np.array_equal(proj_a.transform(raw), proj_c.transform(raw))  # ...the scaler on the rows.
    with pytest.raises(RuntimeError):
        RandomProjection(3, seed=1).transform(raw)


def test_representation_size_metrics(episodes):
    x, p = episodes
    metrics = representation_size_metrics(macro_features(x, p, bins=4))
    assert metrics["n_features"] == 15
    assert 0 < metrics["marginal_entropy_bits"] <= 15 * np.log2(40)
    assert 1 <= metrics["effective_rank_99"] <= 15
    constant = np.hstack([np.ones((40, 1)), np.arange(40)[:, None]])
    assert representation_size_metrics(constant)["n_constant_features"] == 1


def test_seed_data_splits_are_disjoint_and_deterministic():
    a = generate_seed_data(1000, n_train=30, n_val=10, n_test=12, n_floor_realizations=3)
    b = generate_seed_data(1000, n_train=30, n_val=10, n_test=12, n_floor_realizations=3)
    assert a.train.arrays.shape == (30, N) and a.val.arrays.shape == (10, N)
    assert a.test.arrays.shape == (12, N) and a.shift.arrays.shape == (12, N)
    assert np.array_equal(a.train.target, b.train.target)
    assert np.array_equal(a.test.local_target, b.test.local_target)
    assert a.noise_floor["global"]["irreducible_rmse"] == b.noise_floor["global"]["irreducible_rmse"]
    # Splits come from different streams: no identical episodes across splits.
    rows = {tuple(r) for r in a.train.arrays.tolist()}
    assert not any(tuple(r) in rows for r in a.test.arrays.tolist())
    assert not any(tuple(r) in rows for r in a.val.arrays.tolist())
    # Training-pool episodes are nested prefixes across pool sizes. Their
    # *trajectories* are not (the same stream continues after generation), so
    # a run must keep its pool size fixed; the experiments enforce this.
    small = generate_seed_data(1000, n_train=10, n_val=10, n_test=12)
    assert np.array_equal(small.train.arrays, a.train.arrays[:10])
    assert np.array_equal(small.train.success, a.train.success[:10])


def test_seed_spacing_rules():
    validate_seed_spacing([1000, 2000, 3000])
    with pytest.raises(ValueError):
        validate_seed_spacing([1000, 1050])
    with pytest.raises(ValueError):
        validate_seed_spacing([1000, 1000])
    assert max(STREAM_OFFSETS.values()) < 100


def test_noise_floor_is_zero_for_deterministic_dynamics():
    x = np.tile(np.arange(N), (5, 1))
    p = np.ones((5, N), dtype=np.float32)
    mean, sd = conditional_target_statistics(x, p, np.random.default_rng(0), n_realizations=4)
    assert np.all(mean == 0) and np.all(sd == 0)


def test_fit_and_evaluate_is_deterministic_and_reports_costs(episodes):
    x, p = episodes
    y = simulate(x, p, np.random.default_rng(1))
    feats = macro_features(x, p, bins=2)
    sets = [EvalSet("test", feats[20:], y[20:])]
    a = fit_and_evaluate("small", 7, feats[:20], y[:20], sets)
    b = fit_and_evaluate("small", 7, feats[:20], y[:20], sets)
    assert a["test_rmse"] == b["test_rmse"]
    assert a["n_trees"] == CAPACITIES["small"]["max_iter"]
    assert a["total_leaves"] >= a["n_trees"] and a["total_nodes"] >= a["total_leaves"]
    assert a["fit_seconds"] > 0 and a["predict_seconds"] > 0 and a["predict_rows"] == 20
    assert "_model" in a


def test_stats_helpers():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["ci95_low"] < 2.5 < s["ci95_high"]
    d = paired_difference([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert d["mean"] == -1.0 and d["n_a_lower"] == 3 and d["t_pvalue"] < 1e-6
    assert summarize([1.0])["n"] == 1 and np.isnan(summarize([1.0])["sd"])


def test_result_store_resumes(tmp_path):
    store = ResultStore(tmp_path, "demo")
    store.append(1, {"key": "a", "value": 1, "_model": object()})
    store.append(1, {"key": "b", "value": 2})
    loaded = store.load(1)
    assert set(loaded) == {"a", "b"} and "_model" not in loaded["a"]
    store.write_metadata({"x": 1})
    store.write_metadata({"x": 2})
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert len(meta["invocations"]) == 2 and meta["latest"]["config"] == {"x": 2}
    assert "numpy" in meta["latest"]["environment"]


def test_experiment1_end_to_end_is_resumable_and_deterministic(tmp_path):
    args = ["--seeds", "1000", "--budgets", "20", "40", "--capacities", "small", "--restrict",
            "--local-capacities", "small", "--n-val", "10", "--n-test", "15", "--noise-realizations", "2",
            "--bins", "2", "--output", str(tmp_path / "run")]
    experiment1.main(args)
    store = ResultStore(tmp_path / "run", "experiment1")
    first = store.load(1000)
    fits = [r for r in first.values() if r["record_type"] == "fit"]
    assert len(fits) == 2 * 2 * 5          # 2 budgets x 2 tasks x 5 representations
    assert {r["key"] for r in first.values() if r["record_type"] == "noise_floor"} == {"noise_floor|global", "noise_floor|local"}
    # Second invocation must not add records (resumable).
    experiment1.main(args)
    assert len(store.load(1000)) == len(first)
    # A fresh output directory reproduces identical metrics.
    experiment1.main(args[:-1] + [str(tmp_path / "again")])
    second = ResultStore(tmp_path / "again", "experiment1").load(1000)
    for key, record in first.items():
        if record["record_type"] == "fit":
            assert record["test_rmse"] == second[key]["test_rmse"]
    analysis.main(["--experiment1", str(tmp_path / "run"), "--paper-figures", ""])
    summary = pd.read_csv(tmp_path / "run" / "summary" / "exp1_summary.csv")
    assert set(summary.representation) == set(experiment1.REPRESENTATIONS)
    assert (tmp_path / "run" / "figures" / "exp1_learning_curves.png").is_file()
    assert (tmp_path / "run" / "summary" / "tables.md").is_file()


def test_experiment2_end_to_end(tmp_path):
    args = ["--seeds", "1000", "2000", "--bins", "1", "4", "64", "--budgets", "20", "40",
            "--heatmap-capacities", "small", "--heatmap-budgets", "40", "--local-budgets", "40",
            "--importance-bins", "4", "--n-val", "10", "--n-test", "15", "--output", str(tmp_path / "run")]
    experiment2.main(args)
    records = ResultStore(tmp_path / "run", "experiment2").load_all()
    kinds = {r["record_type"] for r in records}
    assert kinds == {"size", "fit", "importance"}
    fits = pd.DataFrame([r for r in records if r["record_type"] == "fit"])
    # Interior rule skips 64 bins; the left rule covers all three resolutions.
    assert set(fits[(fits.feature_set == "macro") & (fits.edge_assignment == "interior")].bins) == {1, 4}
    assert set(fits[(fits.feature_set == "macro") & (fits.edge_assignment == "left")].bins) == {1, 4, 64}
    assert set(fits[fits.feature_set == "random"].n_features) == {6, 15, 195}
    assert (fits[fits.task == "local"].feature_set == "macro").all()
    analysis.main(["--experiment2", str(tmp_path / "run"), "--paper-figures", ""])
    for name in ("exp2_summary.csv", "exp2_size_metrics.csv", "exp2_paired.csv", "exp2_importance.csv"):
        assert (tmp_path / "run" / "summary" / name).is_file()
    assert (tmp_path / "run" / "figures" / "exp2_error_vs_bins.png").is_file()
    assert (tmp_path / "run" / "figures" / "exp2_heatmap.png").is_file()


# ------------------------------------------------------------- Experiment 2b

def test_closure_metrics_exact_matches_give_full_closure():
    """Neighbours with identical conditional means leave no excess variance; random ones leave all of it."""
    from macro_prediction.experiment2b import closure_metrics
    rng = np.random.default_rng(3)
    n, R = 400, 16
    true_mu = rng.normal(size=(n, 1))
    s2 = np.full((n, 1), 0.25)                      # Known realisation variance.
    mu = true_mu + rng.normal(scale=np.sqrt(0.25 / R), size=(n, 1))
    # Pair every configuration with a twin that has the same true mean.
    twins = np.arange(n).reshape(-1, 2)[:, ::-1].ravel()
    mu[twins[::2]] = true_mu[::2] + rng.normal(scale=np.sqrt(0.25 / R), size=(n // 2, 1))
    mu[1::2] = true_mu[::2] + rng.normal(scale=np.sqrt(0.25 / R), size=(n // 2, 1))
    exact = closure_metrics(mu, s2, twins[:, None], np.zeros((n, 1)), R)
    assert abs(exact["excess"]) < 0.05 and exact["closure_index"] > 0.95
    random_nb = rng.integers(0, n, size=(n, 1))
    rand = closure_metrics(mu, s2, random_nb, np.ones((n, 1)), R)
    assert abs(rand["closure_index"]) < 0.25


def test_nearest_distinct_excludes_self_and_duplicates():
    from macro_prediction.experiment2b import nearest_distinct
    z = np.array([[0.0], [0.0], [1.0], [1.0], [5.0], [6.0]])
    micro = np.array([[0], [0], [1], [2], [3], [4]])   # Rows 0 and 1 are the same microstate.
    nb, dist, rel = nearest_distinct(z, micro, 1)
    assert nb[0, 0] not in (0, 1) and nb[1, 0] not in (0, 1)
    assert dist.shape == (6, 1) and 0 < rel < 1
    nb_r, _, _ = nearest_distinct(z, micro, 1, refine_by="random", k0=3, rng=np.random.default_rng(0))
    assert nb_r.shape == (6, 1) and all(nb_r[i, 0] != i for i in range(6))


def test_simulate_futures_shapes_and_sorted_absorbing():
    from macro_prediction.experiment2b import simulate_futures
    x = np.tile(np.arange(N), (3, 1))
    p = np.ones((3, N), dtype=np.float32)
    fut = simulate_futures(x, p, np.random.default_rng(0), horizon=4, n_realizations=2)
    assert fut.shape == (2, 4, 3, 3)
    assert np.all(fut[:, :, :, 0] == 0)   # Sorted arrays stay sorted: zero displacement.


def test_experiment2b_end_to_end(tmp_path):
    from macro_prediction import experiment2b
    args = ["--seeds", "1000", "2000", "--n-configs", "200", "--realizations", "3", "--horizon", "3",
            "--k", "3", "--k0", "8", "--bins", "1", "8", "--random-dims", "3", "27",
            "--output", str(tmp_path / "run")]
    experiment2b.main(args)
    records = ResultStore(tmp_path / "run", "experiment2b").load_all()
    reps = {r["representation"] for r in records}
    assert {"global", "macro-8", "raw", "random-27", "macro-8+random", "macro-8+history", "macro-8+global"} <= reps
    assert {r["n_used"] for r in records if r["representation"] == "raw"} == {50, 100, 200}
    experiment2b.main(args)   # Resumable: nothing new is appended.
    assert len(ResultStore(tmp_path / "run", "experiment2b").load_all()) == len(records)
    analysis.main(["--experiment2b", str(tmp_path / "run"), "--paper-figures", ""])
    assert (tmp_path / "run" / "summary" / "exp2b_summary.csv").is_file()
    assert (tmp_path / "run" / "figures" / "exp2b_closure_vs_horizon.png").is_file()
    assert (tmp_path / "run" / "figures" / "exp2b_refinements.png").is_file()
