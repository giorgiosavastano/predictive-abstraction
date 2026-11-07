"""Model family, capacity presets and timed fitting/evaluation.

All conditions use scikit-learn's ``HistGradientBoostingRegressor``. The
*capacity* presets vary only the number of boosting iterations and the
maximum number of leaves per tree; learning rate, minimum leaf size and L2
regularization are held at the baseline values. ``medium`` is exactly the
baseline configuration, so the baseline numbers are a special case of the
new grid. Early stopping is disabled everywhere so that a preset always uses
its nominal capacity, which keeps "matched estimator settings" meaningful.

Capacity is reported two ways: the nominal setting and the realised model
size (number of trees, total leaves and total nodes), because the number of
leaves actually grown depends on the data and on ``min_samples_leaf``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

SHARED_PARAMS = dict(
    learning_rate=0.075,
    min_samples_leaf=12,
    l2_regularization=2.0,
    early_stopping=False,
)

# Nominal capacity presets (number of trees, maximum leaves per tree).
CAPACITIES: dict[str, dict] = {
    "small": dict(max_iter=50, max_leaf_nodes=4),
    "medium": dict(max_iter=100, max_leaf_nodes=9),     # Baseline setting.
    "large": dict(max_iter=300, max_leaf_nodes=31),
    "xlarge": dict(max_iter=1000, max_leaf_nodes=63),
}
CAPACITY_ORDER = tuple(CAPACITIES)


def make_model(capacity: str, seed: int) -> HistGradientBoostingRegressor:
    if capacity not in CAPACITIES:
        raise ValueError(f"Unknown capacity {capacity!r}; choose from {CAPACITY_ORDER}")
    return HistGradientBoostingRegressor(random_state=seed, **SHARED_PARAMS, **CAPACITIES[capacity])


def model_size(model: HistGradientBoostingRegressor) -> dict:
    """Realised size of a fitted boosting model (parameter-count proxies)."""
    leaves = nodes = 0
    for stage in model._predictors:
        for predictor in stage:
            nodes += int(len(predictor.nodes))
            leaves += int(predictor.nodes["is_leaf"].sum())
    return {"n_trees": int(model.n_iter_), "total_leaves": leaves, "total_nodes": nodes}


def score(y_true: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    return (float(np.sqrt(mean_squared_error(y_true, predictions))),
            float(r2_score(y_true, predictions)))


@dataclass
class EvalSet:
    name: str
    features: np.ndarray
    target: np.ndarray


def fit_and_evaluate(
    capacity: str,
    seed: int,
    train_features: np.ndarray,
    train_target: np.ndarray,
    eval_sets: list[EvalSet],
    *,
    timing_set: str = "test",
) -> dict:
    """Fit one model and return metrics, realised size and wall-clock costs.

    ``fit_seconds`` is the wall-clock time of ``model.fit`` only.
    ``predict_seconds`` is the wall-clock time of predicting the evaluation
    set called ``timing_set`` (normally the 2,000-episode test split).
    Feature-extraction costs are measured separately by the experiment code.
    """
    model = make_model(capacity, seed)
    start = time.perf_counter()
    model.fit(train_features, train_target)
    fit_seconds = time.perf_counter() - start

    record = {
        "capacity": capacity,
        "max_iter": CAPACITIES[capacity]["max_iter"],
        "max_leaf_nodes": CAPACITIES[capacity]["max_leaf_nodes"],
        "n_features": int(train_features.shape[1]),
        "fit_seconds": round(fit_seconds, 5),
    }
    for eval_set in eval_sets:
        start = time.perf_counter()
        predictions = model.predict(eval_set.features)
        elapsed = time.perf_counter() - start
        rmse, r2 = score(eval_set.target, predictions)
        record[f"{eval_set.name}_rmse"] = round(rmse, 6)
        record[f"{eval_set.name}_r2"] = round(r2, 6)
        if eval_set.name == timing_set:
            record["predict_seconds"] = round(elapsed, 6)
            record["predict_rows"] = int(len(eval_set.target))
    record.update(model_size(model))
    record["_model"] = model  # Stripped before persistence; used for importances.
    return record
