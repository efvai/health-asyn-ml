import numpy as np
import pytest

from ml_toolbox.data_loader.windowing import (
    WindowConfig,
    create_label_to_class_map,
    create_windows_for_ml,
)


def _make_dataset():
    # Two classes with enough samples for multiple windows.
    signal_a = np.arange(0, 30, dtype=np.float64).reshape(-1, 1)
    signal_b = np.arange(100, 130, dtype=np.float64).reshape(-1, 1)

    data_list = [signal_a, signal_b]
    metadata_list = [{"class": "a"}, {"class": "b"}]
    return data_list, metadata_list


def test_create_windows_for_ml_uses_class_labels_and_internal_mapping():
    data_list, metadata_list = _make_dataset()

    windows, labels, win_metadata = create_windows_for_ml(
        data_list=data_list,
        metadata_list=metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=False,
    )

    assert windows.shape[0] == 4
    assert labels.shape[0] == 4
    assert len(win_metadata) == 4

    expected_map = {"a": 0, "b": 1}
    for meta, label in zip(win_metadata, labels):
        assert meta["class"] in expected_map
        assert int(label) == expected_map[meta["class"]]


def test_create_windows_for_ml_rejects_removed_target_key_argument():
    data_list, metadata_list = _make_dataset()

    with pytest.raises(TypeError):
        create_windows_for_ml(
            data_list,
            metadata_list,
            window_size=10,
            target_key="class",
        )


def test_create_windows_for_ml_rejects_removed_condition_map_argument():
    data_list, metadata_list = _make_dataset()

    with pytest.raises(TypeError):
        create_windows_for_ml(
            data_list,
            metadata_list,
            window_size=10,
            condition_map={"a": 0, "b": 1},
        )


def test_create_windows_for_ml_random_state_makes_balancing_deterministic():
    data_list, metadata_list = _make_dataset()

    windows_1, labels_1, meta_1 = create_windows_for_ml(
        data_list,
        metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=False,
        random_state=42,
    )

    windows_2, labels_2, meta_2 = create_windows_for_ml(
        data_list,
        metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=False,
        random_state=42,
    )

    assert np.array_equal(windows_1, windows_2)
    assert np.array_equal(labels_1, labels_2)
    assert meta_1 == meta_2


def test_create_windows_for_ml_random_state_makes_shuffle_deterministic():
    data_list, metadata_list = _make_dataset()

    windows_1, labels_1, meta_1 = create_windows_for_ml(
        data_list,
        metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=True,
        random_state=7,
    )

    windows_2, labels_2, meta_2 = create_windows_for_ml(
        data_list,
        metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=True,
        random_state=7,
    )

    assert np.array_equal(windows_1, windows_2)
    assert np.array_equal(labels_1, labels_2)
    assert meta_1 == meta_2


def test_create_label_to_class_map_matches_window_labels():
    data_list, metadata_list = _make_dataset()

    _, labels, win_metadata = create_windows_for_ml(
        data_list=data_list,
        metadata_list=metadata_list,
        window_size=10,
        overlap_ratio=0.5,
        max_windows_per_class=2,
        shuffle=False,
    )

    label_to_class = create_label_to_class_map(win_metadata)

    for meta, label in zip(win_metadata, labels):
        assert label_to_class[int(label)] == meta["class"]


def test_window_config_rejects_removed_padding_argument():
    with pytest.raises(TypeError):
        WindowConfig(window_size=10, step_size=5, padding=True)
