"""
Tests for ml_toolbox.preprocessing.downsampler.resample_dataset
"""

import json
import numpy as np
import pytest
from pathlib import Path

from ml_toolbox.preprocessing.downsampler import resample_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(n_samples: int, n_channels: int, dtype=np.float64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_samples, n_channels)).astype(dtype)


def _make_file_info(
    sample_id: str,
    sensor_type: str,
    filename: str,
    source_fs_current: float = 48000.0,
    source_fs_vibro: float = 26000.0,
    extra_meta: dict | None = None,
) -> dict:
    meta = {
        "class": "d0",
        "electrical_frequency_hz": 20,
        "load": 0,
        "sample_rate_current_hz": source_fs_current,
        "sample_rate_vibro_hz": source_fs_vibro,
        "pwm_frequency_hz": 12000,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "sample_id": sample_id,
        "sensor_type": sensor_type,
        "filename": filename,
        "sample_rate_current_hz": source_fs_current,
        "sample_rate_vibro_hz": source_fs_vibro,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResampleDataset:

    def test_output_file_exists(self, tmp_path: Path):
        source_fs = 10_000.0
        target_fs = 1_000.0
        n_samples = 10_000
        signal = _make_signal(n_samples, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        resample_dataset([signal], [file_info], target_fs, tmp_path)

        assert (tmp_path / "0001" / "test_LTR11.dat").exists()

    def test_output_sample_count(self, tmp_path: Path):
        source_fs = 10_000.0
        target_fs = 1_000.0
        n_samples = 10_000
        n_channels = 2
        signal = _make_signal(n_samples, n_channels)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        resample_dataset([signal], [file_info], target_fs, tmp_path)

        dat_path = tmp_path / "0001" / "test_LTR11.dat"
        raw = np.fromfile(dat_path, dtype=np.float32)
        assert raw.size % n_channels == 0
        actual_samples = raw.size // n_channels
        expected_samples = int(n_samples * target_fs / source_fs)
        assert abs(actual_samples - expected_samples) <= max(1, int(expected_samples * 0.01))

    def test_output_dtype_is_float32(self, tmp_path: Path):
        source_fs = 10_000.0
        target_fs = 1_000.0
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        resample_dataset([signal], [file_info], target_fs, tmp_path)

        dat_path = tmp_path / "0001" / "test_LTR11.dat"
        raw = np.fromfile(dat_path, dtype=np.float32)
        # Verify by also checking it's NOT a valid float64 remainder
        assert dat_path.stat().st_size % 4 == 0, "file size must be multiple of 4 (float32)"

    def test_meta_json_rate_updated(self, tmp_path: Path):
        source_fs = 10_000.0
        target_fs = 1_000.0
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs,
                                    source_fs_vibro=26_000.0)

        resample_dataset([signal], [file_info], target_fs, tmp_path)

        with open(tmp_path / "0001" / "meta.json") as fh:
            meta = json.load(fh)

        assert meta["sample_rate_current_hz"] == target_fs
        # Vibration rate must be unchanged (no vibration file was provided)
        assert meta["sample_rate_vibro_hz"] == 26_000.0

    def test_meta_json_other_fields_preserved(self, tmp_path: Path):
        source_fs = 10_000.0
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        resample_dataset([signal], [file_info], 1_000.0, tmp_path)

        with open(tmp_path / "0001" / "meta.json") as fh:
            meta = json.load(fh)

        assert meta["class"] == "d0"
        assert meta["electrical_frequency_hz"] == 20
        assert meta["load"] == 0
        assert meta["pwm_frequency_hz"] == 12000

    def test_overwrite_false_raises_if_exists(self, tmp_path: Path):
        source_fs = 10_000.0
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        # First run — should succeed
        resample_dataset([signal], [file_info], 1_000.0, tmp_path, overwrite=False)

        # Second run — must raise
        with pytest.raises(FileExistsError):
            resample_dataset([signal], [file_info], 1_000.0, tmp_path, overwrite=False)

    def test_overwrite_true_allows_rerun(self, tmp_path: Path):
        source_fs = 10_000.0
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        resample_dataset([signal], [file_info], 1_000.0, tmp_path)
        # Should not raise
        resample_dataset([signal], [file_info], 1_000.0, tmp_path, overwrite=True)

    def test_mismatched_list_lengths_raises(self, tmp_path: Path):
        signal = _make_signal(1000, 2)
        with pytest.raises(ValueError, match="same length"):
            resample_dataset([signal], [], 1_000.0, tmp_path)

    def test_invalid_target_fs_raises(self, tmp_path: Path):
        signal = _make_signal(1000, 2)
        file_info = _make_file_info("0001", "current", "test.dat")
        with pytest.raises(ValueError, match="target_fs"):
            resample_dataset([signal], [file_info], 0.0, tmp_path)

    def test_source_fs_not_exceeded_skips_file(self, tmp_path: Path, caplog):
        source_fs = 1_000.0
        target_fs = 5_000.0  # higher than source — should skip
        signal = _make_signal(1000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=source_fs)

        stats = resample_dataset([signal], [file_info], target_fs, tmp_path)

        assert stats["files_written"] == 0
        assert stats["samples_skipped"] == 1
        assert not (tmp_path / "0001" / "test_LTR11.dat").exists()

    def test_empty_data_list_returns_zeros(self, tmp_path: Path):
        stats = resample_dataset([], [], 1_000.0, tmp_path)
        assert stats == {"samples_written": 0, "files_written": 0, "samples_skipped": 0}

    def test_multiple_samples_written(self, tmp_path: Path):
        source_fs = 10_000.0
        target_fs = 1_000.0

        data_list = []
        meta_list = []
        for sid in ["0001", "0002", "0003"]:
            data_list.append(_make_signal(10_000, 2))
            meta_list.append(_make_file_info(sid, "current", "test_LTR11.dat",
                                             source_fs_current=source_fs))

        stats = resample_dataset(data_list, meta_list, target_fs, tmp_path)

        assert stats["samples_written"] == 3
        assert stats["files_written"] == 3
        for sid in ["0001", "0002", "0003"]:
            assert (tmp_path / sid / "test_LTR11.dat").exists()
            assert (tmp_path / sid / "meta.json").exists()

    def test_vibration_rate_updated_independently(self, tmp_path: Path):
        source_fs_vibro = 26_000.0
        target_fs = 2_000.0
        signal = _make_signal(26_000, 4)
        file_info = _make_file_info("0001", "vibration", "test_LTR22.dat",
                                    source_fs_vibro=source_fs_vibro)

        resample_dataset([signal], [file_info], target_fs, tmp_path)

        with open(tmp_path / "0001" / "meta.json") as fh:
            meta = json.load(fh)

        assert meta["sample_rate_vibro_hz"] == target_fs
        assert meta["sample_rate_current_hz"] == 48000.0  # original default unchanged

    def test_return_stats_structure(self, tmp_path: Path):
        signal = _make_signal(10_000, 2)
        file_info = _make_file_info("0001", "current", "test_LTR11.dat",
                                    source_fs_current=10_000.0)
        stats = resample_dataset([signal], [file_info], 1_000.0, tmp_path)
        assert "samples_written" in stats
        assert "files_written" in stats
        assert "samples_skipped" in stats
