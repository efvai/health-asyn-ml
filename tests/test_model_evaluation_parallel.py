from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_toolbox.analysis.model_evaluation import cv_shap, evaluate_model_cv


def _build_data() -> tuple[np.ndarray, np.ndarray]:
    features, labels = make_classification(
        n_samples=90,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return features.astype(np.float64), labels.astype(np.int64)


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=30,
                    random_state=42,
                    max_depth=6,
                    min_samples_split=2,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )


def _shap_to_array(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v) for v in shap_values], axis=0)
    return np.asarray(shap_values)


def test_evaluate_model_cv_parallel_matches_serial() -> None:
    features, labels = _build_data()

    serial = evaluate_model_cv(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
    )
    parallel = evaluate_model_cv(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=True,
        n_jobs=2,
    )

    assert np.allclose(serial["cv_scores"], parallel["cv_scores"])
    assert np.array_equal(serial["confusion_matrix"], parallel["confusion_matrix"])
    assert np.array_equal(serial["precision_per_class"], parallel["precision_per_class"])
    assert np.array_equal(serial["recall_per_class"], parallel["recall_per_class"])
    assert np.array_equal(serial["f1_per_class"], parallel["f1_per_class"])

    assert serial["accuracy"] == pytest.approx(parallel["accuracy"])
    assert serial["precision"] == pytest.approx(parallel["precision"])
    assert serial["recall"] == pytest.approx(parallel["recall"])
    assert serial["f1_score"] == pytest.approx(parallel["f1_score"])


def test_cv_shap_parallel_matches_serial_fold_layout() -> None:
    features, labels = _build_data()

    serial = cv_shap(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
    )
    parallel = cv_shap(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=True,
        n_jobs=2,
    )

    serial_folds = serial["shap_values_per_fold"]
    parallel_folds = parallel["shap_values_per_fold"]

    assert len(serial_folds) == len(parallel_folds) == 3

    for fold_serial, fold_parallel in zip(serial_folds, parallel_folds):
        assert np.array_equal(fold_serial["train_idx"], fold_parallel["train_idx"])
        assert np.array_equal(fold_serial["val_idx"], fold_parallel["val_idx"])

        serial_arr = _shap_to_array(fold_serial["shap_values"])
        parallel_arr = _shap_to_array(fold_parallel["shap_values"])

        assert serial_arr.shape == parallel_arr.shape
        assert np.allclose(serial_arr, parallel_arr)


def test_parallel_validation_rejects_invalid_n_jobs() -> None:
    features, labels = _build_data()

    with pytest.raises(ValueError, match="n_jobs must be a positive integer or None"):
        evaluate_model_cv(
            _build_pipeline(),
            features,
            labels,
            cv_folds=3,
            parallel=True,
            n_jobs=0,
        )

    with pytest.raises(ValueError, match="n_jobs must be a positive integer or None"):
        cv_shap(
            _build_pipeline(),
            features,
            labels,
            cv_folds=3,
            parallel=True,
            n_jobs=0,
        )
