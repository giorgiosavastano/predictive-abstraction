"""Representations of the initial system state.

What every representation keeps and discards
--------------------------------------------

*Raw (microscopic) state*, 128 values: the 64 normalized element values
``a_i / 63`` and the 64 edge-reliability values (the last one is padding and
always 1.0). This is the complete observable state; nothing is discarded.

*Global summaries*, 3 values, available at every spatial resolution:

1. mean normalized displacement ``mean_i |a_i - i| / 64``;
2. fraction of adjacent descents ``mean_e [a_e > a_{e+1}]`` over the 63 edges;
3. mean of ``displacement_e * (1 - reliability_e)`` over the 63 edges, an
   interaction between misplaced elements and unreliable exchanges.

*Regional summaries*, 3 values per spatial bin: the 64 positions are divided
into ``bins`` contiguous, nearly equal regions with ``numpy.array_split``.
Each region contributes its mean displacement, its descent fraction and its
mean edge unreliability. Two edge-assignment rules are available:

``"interior"`` (the original extractor): a region only sees the edges that
lie *completely inside* it (``inds[:-1]``), so boundary edges between regions
are not counted by any region. A single-element region has no interior edge,
which is why the original extractor only accepts ``1 <= bins <= 32``.

``"left"``: edge ``e`` (between positions ``e`` and ``e+1``) belongs to the
region containing position ``e``. Every one of the 63 edges is then counted by
exactly one region, and 64 bins become well defined (the last region of a
64-bin split has no edge; its descent/unreliability entries are set to zero).
At 64 bins the regional displacement is ``|a_i - i| / 64`` for every element,
which still differs from the raw state: the *sign* of each displacement and
the element identities are discarded.

Discarded at every resolution below 64 bins: which particular elements are
in which region, the sign and ordering of displacements within a region, and
the exact position of unreliable edges inside a region. Discarded at *every*
resolution including 64: element identities and displacement signs, which is
exactly the information the local counterexample task depends on.

The random projection is a dimension-matched control: a fixed Gaussian matrix
applied to the standardized raw state. It compresses without using knowledge
of the task. Standardization is fitted on the training subset only.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .simulator import N

BINS = 8  # Spatial regions used by the baseline macroscopic representation.
RAW_DIM = 2 * N
GLOBAL_DIM = 3
REGIONAL_PER_BIN = 3
EDGE_ASSIGNMENTS = ("interior", "left")


def _intermediates(arrays: np.ndarray, success: np.ndarray):
    """Shared O(N) building blocks, computed exactly as in the original code."""
    position = np.arange(N)
    displacement = np.abs(arrays.astype(np.float32) - position) / N
    descents = (arrays[:, :-1] > arrays[:, 1:]).astype(np.float32)
    low_reliability = 1.0 - success[:, :-1]
    return displacement, descents, low_reliability


def raw_features(arrays: np.ndarray, success: np.ndarray) -> np.ndarray:
    """Microscopic state: all normalized values plus local exchange probabilities."""
    return np.concatenate((arrays / (N - 1), success), axis=1).astype(np.float32)


def global_features(arrays: np.ndarray, success: np.ndarray) -> np.ndarray:
    """Three array-wide summaries (see module docstring)."""
    displacement, descents, low_reliability = _intermediates(arrays, success)
    return _global_from_intermediates(displacement, descents, low_reliability)


def _global_from_intermediates(displacement, descents, low_reliability):
    return np.column_stack((
        displacement.mean(axis=1),
        descents.mean(axis=1),
        (displacement[:, :-1] * low_reliability).mean(axis=1),
    ))


def _bin_edges(inds: np.ndarray, assignment: str) -> np.ndarray:
    if assignment == "interior":
        return inds[:-1]
    if assignment == "left":
        return inds[inds < N - 1]
    raise ValueError(f"edge_assignment must be one of {EDGE_ASSIGNMENTS}")


def _regional_from_intermediates(displacement, descents, low_reliability, bins, assignment):
    features = []
    for inds in np.array_split(np.arange(N), bins):
        edges = _bin_edges(inds, assignment)
        if len(edges) == 0:
            # Only happens for "left" with 64 bins (last region). Constant
            # zero columns carry no information and keep the layout regular.
            zeros = np.zeros(len(displacement), dtype=np.float32)
            desc, unrel = zeros, zeros
        else:
            desc = descents[:, edges].mean(axis=1)
            unrel = low_reliability[:, edges].mean(axis=1)
        features.append(np.column_stack((displacement[:, inds].mean(axis=1), desc, unrel)))
    return features


def validate_bins(bins: int, edge_assignment: str = "interior") -> None:
    """Raise if ``bins`` is not representable under the given edge rule."""
    if edge_assignment not in EDGE_ASSIGNMENTS:
        raise ValueError(f"edge_assignment must be one of {EDGE_ASSIGNMENTS}")
    upper = N // 2 if edge_assignment == "interior" else N
    if not 1 <= bins <= upper:
        raise ValueError(
            f"bins must be between 1 and {upper} for edge_assignment={edge_assignment!r}"
        )


def macro_features(
    arrays: np.ndarray,
    success: np.ndarray,
    *,
    bins: int = BINS,
    include_global: bool = True,
    include_regional: bool = True,
    edge_assignment: str = "interior",
) -> np.ndarray:
    """Hand-engineered coarse representation with configurable parts.

    With the defaults this reproduces the original macroscopic features
    exactly (``3 + 3 * bins`` columns). ``include_global=False`` gives the
    regional block alone; ``include_regional=False`` gives the three global
    summaries alone (then ``bins`` is irrelevant).
    """
    if not (include_global or include_regional):
        raise ValueError("At least one of global or regional features is required")
    if include_regional:
        validate_bins(bins, edge_assignment)
    displacement, descents, low_reliability = _intermediates(arrays, success)
    parts = []
    if include_global:
        parts.append(_global_from_intermediates(displacement, descents, low_reliability))
    if include_regional:
        parts.extend(_regional_from_intermediates(
            displacement, descents, low_reliability, bins, edge_assignment))
    return np.concatenate(parts, axis=1).astype(np.float32)


def macro_feature_names(
    bins: int = BINS, *, include_global: bool = True, include_regional: bool = True
) -> list[str]:
    names = []
    if include_global:
        names += ["global_displacement", "global_descents", "global_disp_x_unreliability"]
    if include_regional:
        for b in range(bins):
            names += [f"bin{b:02d}_displacement", f"bin{b:02d}_descents", f"bin{b:02d}_unreliability"]
    return names


def representations(arrays: np.ndarray, success: np.ndarray, *, bins: int = BINS):
    """Build full-state, meaningful coarse and minimal scalar representations.

    IMPORTANT: the macro features use only the *initial* snapshot and fault
    profile. No future-state or target information enters this function.
    Each macro feature takes O(N) work; we deliberately avoid expensive
    inversion counting, which would create an unfair resource comparison.

    This is the original baseline entry point and is unchanged in behaviour:
    it accepts ``1 <= bins <= 32`` and returns ``(raw, macro, scalar)``.
    """
    if not 1 <= bins <= N // 2:
        raise ValueError(f"bins must be between 1 and {N // 2}")
    m = len(arrays)
    raw = raw_features(arrays, success)
    displacement, descents, low_reliability = _intermediates(arrays, success)
    global_block = _global_from_intermediates(displacement, descents, low_reliability)
    regional = _regional_from_intermediates(displacement, descents, low_reliability, bins, "interior")
    macro = np.concatenate([global_block, *regional], axis=1).astype(np.float32)
    # Initial disorder alone: simple control. Kept in float64 as in the
    # original code so that the baseline learner sees bit-identical inputs.
    scalar = global_block[:, :1].copy()
    assert raw.shape == (m, RAW_DIM)
    assert macro.shape == (m, GLOBAL_DIM + REGIONAL_PER_BIN * bins)
    return raw, macro, scalar


class RandomProjection:
    """Seeded Gaussian projection of the standardized raw state.

    ``fit`` uses training rows only, so neither the standardization statistics
    nor the projection matrix can see validation or test episodes. The matrix
    entries are scaled by ``1/sqrt(n_components)`` exactly as in the baseline.
    """

    def __init__(self, n_components: int, seed: int):
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.scaler: StandardScaler | None = None
        self.matrix: np.ndarray | None = None

    def fit(self, train_raw: np.ndarray) -> "RandomProjection":
        self.scaler = StandardScaler().fit(train_raw)
        rng = np.random.default_rng(self.seed)
        self.matrix = (rng.normal(size=(train_raw.shape[1], self.n_components))
                       .astype(np.float32) / np.sqrt(self.n_components))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.matrix is None:
            raise RuntimeError("RandomProjection must be fitted before transform")
        return (self.scaler.transform(raw) @ self.matrix).astype(np.float32)


def representation_size_metrics(features: np.ndarray, *, max_bins: int = 255) -> dict:
    """Encoding-cost proxies for a representation, computed on training rows.

    ``n_features``: the plain column count.

    ``marginal_entropy_bits``: each column is discretized into at most
    ``max_bins`` quantile bins (mirroring what the histogram gradient-boosting
    learner does internally), its Shannon entropy in bits is computed, and the
    entropies are summed. By subadditivity this is an *upper bound* on the
    joint entropy of the discretized representation, i.e. on the number of
    bits required to encode what the learner can see. It is NOT a mutual
    information with the target and should not be read as one.

    ``effective_rank_99``: number of principal components of the standardized
    representation needed to explain 99% of its variance (a redundancy proxy).

    ``mean_abs_correlation``: mean absolute off-diagonal Pearson correlation
    between columns (another redundancy proxy; constant columns are skipped).
    """
    x = np.asarray(features, dtype=np.float64)
    n_rows, n_cols = x.shape
    bits = 0.0
    for col in range(n_cols):
        values = x[:, col]
        uniques = np.unique(values)
        if len(uniques) <= max_bins:
            codes = np.searchsorted(uniques, values)
        else:
            quantiles = np.quantile(values, np.linspace(0, 1, max_bins + 1)[1:-1])
            codes = np.searchsorted(quantiles, values)
        counts = np.bincount(codes)
        probs = counts[counts > 0] / n_rows
        bits += float(-(probs * np.log2(probs)).sum())

    std = x.std(axis=0)
    keep = std > 0
    rank = 0
    mean_corr = float("nan")
    if keep.sum() > 0:
        z = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
        singular = np.linalg.svd(z, compute_uv=False)
        explained = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        rank = int(np.searchsorted(explained, 0.99) + 1)
        if keep.sum() > 1:
            corr = np.corrcoef(z, rowvar=False)
            off = corr[~np.eye(len(corr), dtype=bool)]
            mean_corr = float(np.abs(off).mean())
    return {
        "n_features": int(n_cols),
        "n_constant_features": int((~keep).sum()),
        "marginal_entropy_bits": round(bits, 3),
        "effective_rank_99": rank,
        "mean_abs_correlation": None if np.isnan(mean_corr) else round(mean_corr, 4),
    }
