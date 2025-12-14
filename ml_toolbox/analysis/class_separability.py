"""
Class separability analysis utilities based on Cohen's d effect size.

This module provides tiered helpers that compute separability scores on top of
feature matrices and classification labels. The output flows mimic scikit-learn's
metrics → analysis → report pattern:

- ``compute_cohens_d``: low-level numeric calculation on two samples.
- ``cohens_d_summary``: mid-level JSON-friendly summary over all feature/label pairs.
- ``build_separability_report``: high-level pandas ``DataFrame`` ready for export.
- ``plot_separability``: optional plotting hook (imports handled lazily).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray]


def _sanitize_array(values: ArrayLike, *, nan_policy: str = "omit") -> np.ndarray:
    """Convert input to a 1-D float array, optionally dropping NaNs."""
    arr = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        arr = arr[~np.isnan(arr)]
    elif nan_policy != "propagate":
        raise ValueError(f"Unsupported nan_policy='{nan_policy}'")
    return arr


def compute_cohens_d(
    sample_a: ArrayLike,
    sample_b: ArrayLike,
    *,
    nan_policy: str = "omit"
) -> float:
    """Compute Cohen's d effect size between two samples.

    Args:
        sample_a: Numeric observations for the first class.
        sample_b: Numeric observations for the second class.
        nan_policy: ``'omit'`` (default) removes NaNs prior to calculation,
            ``'propagate'`` keeps them, potentially returning ``nan``.

    Returns:
        Cohen's d (float). Returns ``nan`` when the pooled standard deviation is
        not defined (e.g., fewer than two observations per sample).
    """
    a = _sanitize_array(sample_a, nan_policy=nan_policy)
    b = _sanitize_array(sample_b, nan_policy=nan_policy)

    if a.size < 2 or b.size < 2:
        return float("nan")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    pooled_denom = ((a.size - 1) * var_a + (b.size - 1) * var_b)
    pooled_denom /= (a.size + b.size - 2)

    if pooled_denom <= 0:
        # If both variances are zero, samples are constant; separability is only
        # defined when means are identical. Otherwise effect size tends to infinity.
        if pooled_denom == 0:
            if mean_a == mean_b:
                return 0.0
            return float("inf") if mean_a > mean_b else float("-inf")
        # Negative values indicate numerical issues, fall back to NaN.
        return float("nan")

    pooled_std = float(np.sqrt(pooled_denom))
    return (mean_a - mean_b) / pooled_std


def _prepare_class_pairs(
    labels: np.ndarray,
    *,
    class_pairs: Optional[Iterable[Tuple[Any, Any]]] = None
) -> List[Tuple[Any, Any]]:
    """Determine class pairs for separability analysis."""
    unique_labels = np.unique(labels)

    if class_pairs is not None:
        return [(a, b) for a, b in class_pairs]

    if unique_labels.size < 2:
        raise ValueError("At least two distinct classes are required for Cohen's d")

    if unique_labels.size == 2:
        return [(unique_labels[0], unique_labels[1])]

    return list(combinations(unique_labels, 2))


def cohens_d_summary(
    features: np.ndarray,
    labels: Sequence[Any],
    feature_names: Optional[Sequence[str]] = None,
    *,
    class_pairs: Optional[Iterable[Tuple[Any, Any]]] = None,
    nan_policy: str = "omit"
) -> List[Dict[str, Any]]:
    """Compute Cohen's d for every feature across specified class pairs.

    Args:
        features: ``(n_samples, n_features)`` feature matrix.
        labels: Sequence of class labels (length ``n_samples``).
        feature_names: Optional feature name list; falls back to indexed names.
        class_pairs: Optional iterable of class-pair tuples; defaults to binary
            pair or all pairwise combinations for multi-class problems.
        nan_policy: How to handle NaNs before computing Cohen's d.

    Returns:
        A list of dictionaries, each JSON-safe, with keys:
        ``feature``, ``feature_index``, ``class_pair``, ``class_a``, ``class_b``,
        ``d``, ``mean_c1``, ``mean_c2``, ``std_c1``, ``std_c2``, ``n_c1``, ``n_c2``.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2-D array")

    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != features.shape[0]:
        raise ValueError("labels length must match number of samples")

    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(features.shape[1])]
    elif len(feature_names) != features.shape[1]:
        raise ValueError("feature_names length must match number of features")

    class_pairs_list = _prepare_class_pairs(labels_arr, class_pairs=class_pairs)
    results: List[Dict[str, Any]] = []

    for pair_idx, (class_a, class_b) in enumerate(class_pairs_list):
        mask_a = labels_arr == class_a
        mask_b = labels_arr == class_b

        if mask_a.sum() < 2 or mask_b.sum() < 2:
            # Not enough samples for reliable statistics; skip the pair entirely.
            continue

        for feat_idx, feat_name in enumerate(feature_names):
            values_a = _sanitize_array(features[mask_a, feat_idx], nan_policy=nan_policy)
            values_b = _sanitize_array(features[mask_b, feat_idx], nan_policy=nan_policy)

            if values_a.size < 2 or values_b.size < 2:
                effect_size = float("nan")
                mean_a = mean_b = std_a = std_b = float("nan")
            else:
                effect_size = compute_cohens_d(values_a, values_b, nan_policy="propagate")
                mean_a = float(np.mean(values_a))
                mean_b = float(np.mean(values_b))
                std_a = float(np.std(values_a, ddof=1)) if values_a.size > 1 else 0.0
                std_b = float(np.std(values_b, ddof=1)) if values_b.size > 1 else 0.0

            class_pair_label = f"{class_a} vs {class_b}"
            results.append({
                "feature": feat_name,
                "feature_index": feat_idx,
                "class_pair": class_pair_label,
                "class_a": class_a,
                "class_b": class_b,
                "d": effect_size,
                "mean_c1": mean_a,
                "mean_c2": mean_b,
                "std_c1": std_a,
                "std_c2": std_b,
                "n_c1": int(values_a.size),
                "n_c2": int(values_b.size),
                "pair_index": pair_idx,
            })

    return results


def build_separability_report(
    summary: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    features: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
    feature_names: Optional[Sequence[str]] = None,
    class_pairs: Optional[Iterable[Tuple[Any, Any]]] = None,
    nan_policy: str = "omit",
    sort_by: str = "d",
    ascending: bool = False
) -> pd.DataFrame:
    """Assemble a pandas DataFrame from Cohen's d results.

    Args:
        summary: Output from :func:`cohens_d_summary`. If omitted, ``features`` and
            ``labels`` must be provided and the summary will be computed on the fly.
        features: Feature matrix used to compute statistics when ``summary`` is ``None``.
        labels: Class labels corresponding to ``features`` when ``summary`` is ``None``.
        feature_names: Optional feature names consumed when computing a summary.
        class_pairs: Optional class-pair iterable forwarded to :func:`cohens_d_summary`.
        nan_policy: Passed through to :func:`cohens_d_summary` when needed.
        sort_by: Column name to sort the report by (defaults to ``'d'``).
        ascending: Sort order flag.

    Returns:
        ``pd.DataFrame`` containing at least the columns
        ``['feature', 'class_pair', 'd', 'mean_c1', 'mean_c2', 'std_c1', 'std_c2']``.
    """
    if summary is None:
        if features is None or labels is None:
            raise ValueError("Either provide summary or supply features and labels")
        summary = cohens_d_summary(
            features,
            labels,
            feature_names,
            class_pairs=class_pairs,
            nan_policy=nan_policy,
        )

    df = pd.DataFrame(summary)
    if df.empty:
        return df

    required_columns = [
        "feature",
        "class_pair",
        "d",
        "mean_c1",
        "mean_c2",
        "std_c1",
        "std_c2",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    df = df[required_columns + [col for col in df.columns if col not in required_columns]]

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return df


def plot_separability(
    df: pd.DataFrame,
    *,
    backend: str = "matplotlib",
    top_n: int = 20,
    absolute: bool = True,
    **kwargs: Any
) -> Any:
    """Optional plotting hook for separability reports.

    Args:
        df: DataFrame produced by :func:`build_separability_report`.
        backend: Plotting backend identifier (currently supports ``'matplotlib'``).
        top_n: Number of rows to plot, ranked by ``|d|`` when ``absolute`` is True.
        absolute: Plot the absolute effect size magnitude when True.
        **kwargs: Forwarded to the backend-specific plotting call.

    Returns:
        Backend-dependent object (e.g., Matplotlib ``Axes``). Returns ``None`` if
        the backend is not recognised.
    """
    if df.empty:
        raise ValueError("Separability DataFrame is empty; nothing to plot")

    data = df.copy()
    metric = data["d"].abs() if absolute else data["d"]
    data = data.assign(_metric=metric).sort_values("_metric", ascending=False).head(top_n)

    if backend.lower() == "matplotlib":
        import matplotlib.pyplot as plt  # type: ignore  # Lazy import to keep dependencies optional

        ax = kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))

        ax.barh(
            y=[f"{row.class_pair} | {row.feature}" for row in data.itertuples()],
            width=data["_metric"],
            color=kwargs.pop("color", "tab:blue"),
            alpha=kwargs.pop("alpha", 0.8),
        )
        ax.invert_yaxis()
        ylabel = "|Cohen's d|" if absolute else "Cohen's d"
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Class Pair | Feature")
        ax.set_title(kwargs.pop("title", "Class Separability (Cohen's d)"))
        return ax

    raise ValueError(f"Unsupported plotting backend '{backend}'")
