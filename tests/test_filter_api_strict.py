from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_toolbox.data_loader.data_loader import DataLoader
from ml_toolbox.data_loader.dataset_manager import DatasetManager


def _write_sample(
    root: Path,
    sample_id: str,
    *,
    cls: str,
    load: float,
    freq: float,
    current_hz: float = 10_000.0,
    vibro_hz: float = 10_000.0,
    pwm_hz: float = 5_000.0,
) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "class": cls,
        "electrical_frequency_hz": freq,
        "load": load,
        "sample_rate_current_hz": current_hz,
        "sample_rate_vibro_hz": vibro_hz,
        "pwm_frequency_hz": pwm_hz,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Empty files are sufficient for indexing.
    (sample_dir / "test_LTR11_001.dat").write_text("", encoding="utf-8")
    (sample_dir / "test_LTR22_001.dat").write_text("", encoding="utf-8")


@pytest.fixture
def manager(tmp_path: Path) -> DatasetManager:
    _write_sample(
        tmp_path,
        "0001",
        cls="System Misalignment",
        load=0.0,
        freq=20.0,
    )
    _write_sample(
        tmp_path,
        "0002",
        cls="healthy",
        load=1.0,
        freq=30.0,
    )
    return DatasetManager(tmp_path)


def test_filter_files_rejects_unknown_class_with_helpful_message(manager: DatasetManager) -> None:
    with pytest.raises(ValueError, match="Unknown classes"):
        manager.filter_files(classes="system_misalignment")


def test_filter_files_rejects_unknown_sensor_type(manager: DatasetManager) -> None:
    with pytest.raises(ValueError, match="Unknown sensor_types"):
        manager.filter_files(sensor_types="acoustic")


def test_filter_files_accepts_new_names_and_list_values(manager: DatasetManager) -> None:
    files = manager.filter_files(
        classes=["System Misalignment", "healthy"],
        loads=[0.0, 1.0],
        frequencies=[20.0, 30.0],
        sensor_types=["current", "vibration"],
        sample_ids=["0001", "0002"],
    )

    assert len(files) == 4


def test_filter_files_legacy_aliases_are_removed(manager: DatasetManager) -> None:
    with pytest.raises(TypeError):
        manager.filter_files(condition="healthy")


def test_load_batch_legacy_aliases_are_removed(tmp_path: Path) -> None:
    _write_sample(
        tmp_path,
        "0001",
        cls="healthy",
        load=0.0,
        freq=20.0,
    )
    loader = DataLoader(tmp_path)

    with pytest.raises(TypeError):
        loader.load_batch(sensor_type="current")
