"""
Model evaluation utilities extracted from `cv_analysis`.

Provides `evaluate_model_cv(features, labels, cv_folds=5)` that performs
cross-validation and returns a results dictionary. This version does not
accept or include a `frequency` argument so it can be reused across modules.
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def _resolve_n_jobs(cv_folds: int, n_jobs: int | None) -> int:
    if n_jobs is None:
        return max(1, min(cv_folds, os.cpu_count() or 1))
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or None, got {n_jobs!r}.")
    return n_jobs


def _resolve_groups(
    labels: np.ndarray,
    groups: np.ndarray | None,
    win_metadata: List[dict] | None,
    group_by: str,
) -> np.ndarray | None:
    if groups is not None:
        groups_arr = np.asarray(groups)
        if groups_arr.ndim != 1:
            raise ValueError(f"groups must be 1D, got shape {groups_arr.shape}.")
        if groups_arr.shape[0] != labels.shape[0]:
            raise ValueError(
                f"groups length ({groups_arr.shape[0]}) must match labels length ({labels.shape[0]})."
            )
        return groups_arr

    if win_metadata is None:
        return None

    if group_by not in {"sample_id", "path"}:
        raise ValueError(
            f"group_by must be 'sample_id' or 'path', got {group_by!r}."
        )

    if len(win_metadata) != labels.shape[0]:
        raise ValueError(
            f"win_metadata length ({len(win_metadata)}) must match labels length ({labels.shape[0]})."
        )

    resolved_groups: List[str] = []
    for idx, meta in enumerate(win_metadata):
        if not isinstance(meta, dict):
            raise ValueError(
                f"win_metadata[{idx}] must be a dict, got {type(meta).__name__}."
            )

        if group_by == "sample_id":
            group_value = meta.get("sample_id")
            if group_value is None:
                raise ValueError(
                    f"win_metadata[{idx}] missing 'sample_id' required for grouped CV."
                )
        else:
            group_value = meta.get("path", meta.get("absolute_path"))
            if group_value is None:
                raise ValueError(
                    f"win_metadata[{idx}] missing 'path'/'absolute_path' required for grouped CV."
                )

        resolved_groups.append(str(group_value))

    return np.asarray(resolved_groups, dtype=object)


def _build_fold_splits(
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    groups: np.ndarray | None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        n_unique_groups = np.unique(groups).size
        if n_unique_groups < cv_folds:
            raise ValueError(
                f"Grouped CV requires at least cv_folds unique groups, got {n_unique_groups} groups for cv_folds={cv_folds}."
            )
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        return list(cv.split(features, labels, groups))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return list(cv.split(features, labels))


def _run_cv_fold(
    fold_idx: int,
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
):
    X_train, X_val = features[train_idx], features[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    est = clone(pipeline)
    est.fit(X_train, y_train)

    preds = est.predict(X_val)
    score = est.score(X_val, y_val)

    return {
        "fold_idx": fold_idx,
        "val_idx": val_idx,
        "preds": preds,
        "score": score,
        "estimator": est,
    }


def _run_cv_shap_fold(
    fold_idx: int,
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
):
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]

    est = clone(pipeline)
    est.fit(X_train, y_train)

    if "scaler" not in est.named_steps or "rf" not in est.named_steps:
        raise ValueError(
            "cv_shap requires pipeline.named_steps to include both 'scaler' and 'rf'."
        )

    scaler = est.named_steps["scaler"]
    model = est.named_steps["rf"]

    X_val_scaled = scaler.transform(X_val)

    import shap  # lazy import to avoid loading numba/llvmlite at module import time

    # Use tree-path-dependent mode for exact tree-model additivity.
    # Passing a background matrix can trigger interventional estimates that
    # intermittently fail strict additivity checks for RF classifiers.
    try:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
    except TypeError:
        # Backward compatibility for older SHAP versions.
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

    shap_vals = explainer.shap_values(X_val_scaled)

    return {
        "fold_idx": fold_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "shap_values": shap_vals,
        "expected_value": explainer.expected_value,
        "estimator": est,
    }

def evaluate_model_cv(
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    parallel: bool = False,
    n_jobs: int | None = None,
    groups: np.ndarray | None = None,
    win_metadata: List[dict] | None = None,
    group_by: str = "sample_id",
) -> Dict[str, Any]:
    """
    Evaluate model performance using cross-validation (standalone).

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels (n_samples,)
        cv_folds: Number of CV folds (default: 5)

    Returns:
        Dictionary with CV results including accuracy, F1, precision, recall, confusion matrix,
        per-class metrics, cv scores and simple dataset info.
    """
    print(f"Evaluating model (features: {features.shape}, samples: {len(labels)})...")

    cv_data = cross_validate_with_models(
        pipeline,
        features,
        labels,
        cv_folds=cv_folds,
        parallel=parallel,
        n_jobs=n_jobs,
        groups=groups,
        win_metadata=win_metadata,
        group_by=group_by,
    )
    cv_predictions = cv_data["cv_predictions"]
    cv_scores = cv_data["cv_scores"]

    # Determine if multi-class or binary
    unique_labels = np.unique(labels)
    is_multiclass = len(unique_labels) > 2
    average_method = 'weighted' if is_multiclass else 'binary'
    pos_label = None if is_multiclass else unique_labels[-1]

    # Calculate overall metrics
    accuracy = accuracy_score(labels, cv_predictions)
    metric_kwargs = {
        "average": average_method,
        "zero_division": 0,
    }
    if pos_label is not None:
        metric_kwargs["pos_label"] = pos_label

    precision = precision_score(labels, cv_predictions, **metric_kwargs)
    recall = recall_score(labels, cv_predictions, **metric_kwargs)
    f1 = f1_score(labels, cv_predictions, **metric_kwargs)

    # Class-wise metrics
    precision_per_class = precision_score(
        labels,
        cv_predictions,
        average=None,
        zero_division=0,
        labels=unique_labels,
    )
    recall_per_class = recall_score(
        labels,
        cv_predictions,
        average=None,
        zero_division=0,
        labels=unique_labels,
    )
    f1_per_class = f1_score(
        labels,
        cv_predictions,
        average=None,
        zero_division=0,
        labels=unique_labels,
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(labels, cv_predictions, labels=unique_labels)

    # Label distribution
    label_counts = np.unique(labels, return_counts=True)
    label_distribution = dict(zip(label_counts[0], label_counts[1]))

    results = {
        'cv_scores': cv_scores,
        'mean_accuracy': float(cv_scores.mean()),
        'std_accuracy': float(cv_scores.std()),
        'best_fold': float(cv_scores.max()),
        'worst_fold': float(cv_scores.min()),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'label_distribution': label_distribution,
        'n_samples': int(len(labels)),
        'n_features': int(features.shape[1]) if features.ndim > 1 else 1,
        'unique_labels': unique_labels.tolist()
    }

    print(f"Mean CV Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}, F1: {results['f1_score']:.3f}")

    return results

def cross_validate_with_models(
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    parallel: bool = False,
    n_jobs: int | None = None,
    groups: np.ndarray | None = None,
    win_metadata: List[dict] | None = None,
    group_by: str = "sample_id",
):
    """
    Perform cross-validation manually and return:
    - cv_scores
    - cv_predictions (OOF predictions)
    - estimators = [(model, val_indices), ...] for each fold
    """
    resolved_groups = _resolve_groups(
        labels=labels,
        groups=groups,
        win_metadata=win_metadata,
        group_by=group_by,
    )
    n = len(labels)

    cv_predictions = np.empty(n, dtype=labels.dtype)
    cv_scores = []
    estimators = []

    fold_splits = _build_fold_splits(
        features=features,
        labels=labels,
        cv_folds=cv_folds,
        groups=resolved_groups,
    )

    if parallel and len(fold_splits) > 1:
        workers = _resolve_n_jobs(cv_folds=cv_folds, n_jobs=n_jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_cv_fold,
                    fold_idx=fold_idx,
                    pipeline=pipeline,
                    features=features,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )
                for fold_idx, (train_idx, val_idx) in enumerate(fold_splits)
            ]
            fold_results = [f.result() for f in futures]
        fold_results.sort(key=lambda item: item["fold_idx"])
        for fold_result in fold_results:
            val_idx = fold_result["val_idx"]
            cv_predictions[val_idx] = fold_result["preds"]
            cv_scores.append(fold_result["score"])
            estimators.append((fold_result["estimator"], val_idx))
    else:
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            fold_result = _run_cv_fold(
                fold_idx=fold_idx,
                pipeline=pipeline,
                features=features,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
            )
            cv_predictions[val_idx] = fold_result["preds"]
            cv_scores.append(fold_result["score"])
            estimators.append((fold_result["estimator"], val_idx))

    return {
        "cv_scores": np.array(cv_scores),
        "cv_predictions": cv_predictions,
        "estimators": estimators
    }

def cv_shap(
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    parallel: bool = False,
    n_jobs: int | None = None,
    groups: np.ndarray | None = None,
    win_metadata: List[dict] | None = None,
    group_by: str = "sample_id",
):
    resolved_groups = _resolve_groups(
        labels=labels,
        groups=groups,
        win_metadata=win_metadata,
        group_by=group_by,
    )

    shap_per_fold = []
    estimators = []

    fold_splits = _build_fold_splits(
        features=features,
        labels=labels,
        cv_folds=cv_folds,
        groups=resolved_groups,
    )

    if parallel and len(fold_splits) > 1:
        workers = _resolve_n_jobs(cv_folds=cv_folds, n_jobs=n_jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_cv_shap_fold,
                    fold_idx=fold_idx,
                    pipeline=pipeline,
                    features=features,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )
                for fold_idx, (train_idx, val_idx) in enumerate(fold_splits)
            ]
            fold_results = [f.result() for f in futures]
        fold_results.sort(key=lambda item: item["fold_idx"])
        for fold_result in fold_results:
            estimators.append((fold_result["estimator"], fold_result["val_idx"]))
            shap_per_fold.append({
                "train_idx": fold_result["train_idx"],
                "val_idx": fold_result["val_idx"],
                "shap_values": fold_result["shap_values"],
                "expected_value": fold_result["expected_value"],
            })
    else:
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            fold_result = _run_cv_shap_fold(
                fold_idx=fold_idx,
                pipeline=pipeline,
                features=features,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
            )
            estimators.append((fold_result["estimator"], fold_result["val_idx"]))
            shap_per_fold.append({
                "train_idx": fold_result["train_idx"],
                "val_idx": fold_result["val_idx"],
                "shap_values": fold_result["shap_values"],
                "expected_value": fold_result["expected_value"],
            })

    return {
        "shap_values_per_fold": shap_per_fold,
        "estimators": estimators
    }
