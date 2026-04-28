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


def _build_grouped_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.RandomState(42)
    n_groups = 30
    samples_per_group = 3
    n_samples = n_groups * samples_per_group
    n_features = 8

    groups = np.repeat(np.arange(n_groups), samples_per_group)
    labels = (groups % 2).astype(np.int64)

    features = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    features[:, 0] += labels * 1.5
    features[:, 1] += groups * 0.01

    win_metadata = [
        {
            "sample_id": f"{int(group):04d}",
            "path": f"{int(group):04d}/sensor_a.dat",
        }
        for group in groups
    ]

    return features.astype(np.float64), labels, groups, win_metadata


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


def test_evaluate_model_cv_grouped_parallel_matches_serial() -> None:
    features, labels, groups, _ = _build_grouped_data()

    serial = evaluate_model_cv(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
        groups=groups,
    )
    parallel = evaluate_model_cv(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=True,
        n_jobs=2,
        groups=groups,
    )

    assert np.allclose(serial["cv_scores"], parallel["cv_scores"])
    assert np.array_equal(serial["confusion_matrix"], parallel["confusion_matrix"])
    assert serial["accuracy"] == pytest.approx(parallel["accuracy"])


def test_cv_shap_grouped_splits_keep_groups_disjoint() -> None:
    features, labels, groups, _ = _build_grouped_data()

    shap_result = cv_shap(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
        groups=groups,
    )

    for fold in shap_result["shap_values_per_fold"]:
        train_groups = set(groups[np.asarray(fold["train_idx"])].tolist())
        val_groups = set(groups[np.asarray(fold["val_idx"])].tolist())
        assert train_groups.isdisjoint(val_groups)


def test_cv_shap_uses_tree_path_dependent_tree_explainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = _build_data()
    calls: dict[str, object] = {}

    class DummyTreeExplainer:
        def __init__(self, model, *args, **kwargs):
            calls["kwargs"] = dict(kwargs)
            calls.setdefault("init_calls", 0)
            calls["init_calls"] = int(calls["init_calls"]) + 1
            self.expected_value = np.array([0.5, 0.5], dtype=np.float64)

        def shap_values(self, X, check_additivity=True):
            calls.setdefault("check_additivity", [])
            calls["check_additivity"].append(bool(check_additivity))
            n_samples, n_features = np.asarray(X).shape
            return np.zeros((n_samples, n_features, 2), dtype=np.float64)

    monkeypatch.setattr(
        "ml_toolbox.analysis.model_evaluation.shap.TreeExplainer",
        DummyTreeExplainer,
    )

    result = cv_shap(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
    )

    assert len(result["shap_values_per_fold"]) == 3

    explainer_kwargs = calls["kwargs"]
    assert "data" not in explainer_kwargs
    assert explainer_kwargs.get("feature_perturbation") == "tree_path_dependent"
    assert explainer_kwargs.get("model_output") == "raw"

    check_additivity_calls = calls["check_additivity"]
    assert len(check_additivity_calls) == 3
    assert all(check_additivity_calls)


def test_evaluate_model_cv_can_derive_groups_from_win_metadata() -> None:
    features, labels, _groups, win_metadata = _build_grouped_data()

    result = evaluate_model_cv(
        _build_pipeline(),
        features,
        labels,
        cv_folds=3,
        parallel=False,
        win_metadata=win_metadata,
        group_by="sample_id",
    )

    assert result["n_samples"] == features.shape[0]


def test_grouped_cv_validation_errors() -> None:
    features, labels, groups, win_metadata = _build_grouped_data()

    with pytest.raises(ValueError, match="group_by must be 'sample_id' or 'path'"):
        evaluate_model_cv(
            _build_pipeline(),
            features,
            labels,
            cv_folds=3,
            win_metadata=win_metadata,
            group_by="filename",
        )

    broken_meta = [dict(m) for m in win_metadata]
    broken_meta[0].pop("sample_id", None)
    with pytest.raises(ValueError, match="missing 'sample_id'"):
        evaluate_model_cv(
            _build_pipeline(),
            features,
            labels,
            cv_folds=3,
            win_metadata=broken_meta,
            group_by="sample_id",
        )

    with pytest.raises(ValueError, match="at least cv_folds unique groups"):
        evaluate_model_cv(
            _build_pipeline(),
            features,
            labels,
            cv_folds=3,
            groups=np.zeros_like(groups),
        )
