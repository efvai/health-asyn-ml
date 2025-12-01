"""
Model evaluation utilities extracted from `cv_analysis`.

Provides `evaluate_model_cv(features, labels, cv_folds=5)` that performs
cross-validation and returns a results dictionary. This version does not
accept or include a `frequency` argument so it can be reused across modules.
"""

import numpy as np
from typing import Any, Dict

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


import shap

def evaluate_model_cv(pipeline, features: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
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

    cv_data = cross_validate_with_models(pipeline, features, labels, cv_folds)
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
    cv_folds: int = 5
):
    """
    Perform cross-validation manually and return:
    - cv_scores
    - cv_predictions (OOF predictions)
    - estimators = [(model, val_indices), ...] for each fold
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    n = len(labels)

    cv_predictions = np.empty(n, dtype=labels.dtype)
    cv_scores = []
    estimators = []

    for train_idx, val_idx in cv.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        est = clone(pipeline)
        est.fit(X_train, y_train)

        preds = est.predict(X_val)
        cv_predictions[val_idx] = preds
        cv_scores.append(est.score(X_val, y_val))

        estimators.append((est, val_idx))

    return {
        "cv_scores": np.array(cv_scores),
        "cv_predictions": cv_predictions,
        "estimators": estimators
    }

def cv_shap(
    pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5
):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    shap_per_fold = []
    estimators = []

    for train_idx, val_idx in cv.split(features, labels):
        X_train = features[train_idx]
        y_train = labels[train_idx]

        X_val = features[val_idx]

        # --- 1. clone and fit full pipeline  ---
        est = clone(pipeline)
        est.fit(X_train, y_train)

        # Store for user
        estimators.append((est, val_idx))

        # --- 2. Extract steps from pipeline  ---
        scaler = est.named_steps["scaler"]
        model = est.named_steps["rf"]
        
        # --- 3. Transform train + val exactly as during training ---
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # --- 4. Build SHAP explainer with training fold as background ---
        explainer = shap.TreeExplainer(model, data=X_train_scaled)

        # --- 5. Compute shap values for validation samples (OOF) ---
        shap_vals = explainer.shap_values(X_val_scaled)

        shap_per_fold.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "shap_values": shap_vals,
            "expected_value": explainer.expected_value
        })

    return {
        "shap_values_per_fold": shap_per_fold,
        "estimators": estimators
    }
