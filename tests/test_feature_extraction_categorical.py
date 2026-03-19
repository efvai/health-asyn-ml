import numpy as np
import pytest

from ml_toolbox.data_loader.feature_extraction import (
    FeatureConfig,
    extract_categorical_features,
    extract_features_for_ml,
)


def test_extract_categorical_features_numeric_schema_with_sensor_one_hot():
    metadata_list = [
        {
            "class": "healthy",
            "electrical_frequency_hz": 30.0,
            "load": 0.1,
            "sensor_type": "current",
        },
        {
            "class": "faulty",
            "electrical_frequency_hz": 40.0,
            "load": 0.6,
            "sensor_type": "vibration",
        },
    ]

    categorical_matrix, feature_names = extract_categorical_features(metadata_list)

    assert feature_names == ["frequency_hz", "load", "sensor_current", "sensor_vibration"]
    assert categorical_matrix.shape == (2, 4)

    np.testing.assert_allclose(categorical_matrix[:, 0], np.array([30.0, 40.0]))
    np.testing.assert_allclose(categorical_matrix[:, 1], np.array([0.1, 0.6]))
    np.testing.assert_allclose(categorical_matrix[:, 2], np.array([1.0, 0.0]))
    np.testing.assert_allclose(categorical_matrix[:, 3], np.array([0.0, 1.0]))


def test_extract_categorical_features_single_sensor_has_no_one_hot_columns():
    metadata_list = [
        {"class": "healthy", "electrical_frequency_hz": 20.0, "load": 0.0, "sensor_type": "current"},
        {"class": "healthy", "electrical_frequency_hz": 30.0, "load": 0.2, "sensor_type": "current"},
    ]

    categorical_matrix, feature_names = extract_categorical_features(metadata_list)

    assert feature_names == ["frequency_hz", "load"]
    assert categorical_matrix.shape == (2, 2)
    np.testing.assert_allclose(categorical_matrix[:, 0], np.array([20.0, 30.0]))
    np.testing.assert_allclose(categorical_matrix[:, 1], np.array([0.0, 0.2]))


def test_extract_categorical_features_raises_for_missing_electrical_frequency_hz():
    metadata_list = [{"class": "healthy", "load": 0.1, "sensor_type": "current"}]

    with pytest.raises(ValueError, match="electrical_frequency_hz"):
        extract_categorical_features(metadata_list)


def test_extract_categorical_features_raises_for_non_numeric_load():
    metadata_list = [
        {
            "class": "healthy",
            "electrical_frequency_hz": 30.0,
            "load": "under_load",
            "sensor_type": "current",
        }
    ]

    with pytest.raises(ValueError, match="load"):
        extract_categorical_features(metadata_list)


def test_extract_features_for_ml_appends_categorical_columns():
    windows = np.random.randn(3, 1024, 1)
    metadata_list = [
        {"class": "healthy", "electrical_frequency_hz": 30.0, "load": 0.1, "sensor_type": "current"},
        {"class": "healthy", "electrical_frequency_hz": 30.0, "load": 0.2, "sensor_type": "current"},
        {"class": "healthy", "electrical_frequency_hz": 30.0, "load": 0.3, "sensor_type": "current"},
    ]
    config = FeatureConfig.for_sensor("vibration")

    feature_matrix, feature_names = extract_features_for_ml(
        windows,
        sensor_type="vibration",
        feature_config=config,
        metadata_list=metadata_list,
    )

    assert feature_names[-2:] == ["frequency_hz", "load"]
    assert feature_matrix.shape[1] == len(feature_names)

    np.testing.assert_allclose(feature_matrix[:, -2], np.array([30.0, 30.0, 30.0]))
    np.testing.assert_allclose(feature_matrix[:, -1], np.array([0.1, 0.2, 0.3]))
