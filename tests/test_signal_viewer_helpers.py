from __future__ import annotations

import numpy as np
import pytest

from ml_toolbox.signal_processing import prepare_time_frequency_view


def test_prepare_time_frequency_view_uses_sensor_sampling_rate_and_first_channel() -> None:
    sample = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [0.0, 12.0],
            [-1.0, 13.0],
        ],
        dtype=float,
    )
    metadata = {
        "sensor_type": "vibration",
        "sample_rate_vibro_hz": 2000.0,
        "sample_rate_current_hz": 1000.0,
        "sample_id": "0001",
    }

    view = prepare_time_frequency_view(sample, metadata, channel_index=0, normalize=False)

    np.testing.assert_array_equal(view["channel_signal"], sample[:, 0])
    np.testing.assert_allclose(view["time_axis"], np.array([0.0, 0.0005, 0.0010, 0.0015]))
    assert view["sampling_rate"] == 2000.0
    assert view["spectrum"]["sampling_rate"] == 2000.0
    np.testing.assert_array_equal(view["spectrum"]["freqs"], np.array([0.0, 500.0]))


def test_prepare_time_frequency_view_rejects_invalid_channel_index() -> None:
    sample = np.ones((8, 2), dtype=float)
    metadata = {
        "sensor_type": "current",
        "sample_rate_current_hz": 1000.0,
    }

    with pytest.raises(IndexError, match="channel_index"):
        prepare_time_frequency_view(sample, metadata, channel_index=3)


def test_prepare_time_frequency_view_requires_sensor_sampling_rate() -> None:
    sample = np.ones((8, 2), dtype=float)
    metadata = {
        "sensor_type": "vibration",
    }

    with pytest.raises(KeyError, match="sample_rate_vibro_hz"):
        prepare_time_frequency_view(sample, metadata)