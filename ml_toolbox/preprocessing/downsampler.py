"""
Dataset resampler: rebuild a dataset on disk at a lower sampling rate.
"""

import json
import logging
import copy
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def _rational_ratio(source_fs: float, target_fs: float) -> Tuple[int, int]:
    """Return (up, down) integers for resample_poly via rational approximation."""
    frac = Fraction(target_fs / source_fs).limit_denominator(1000)
    return frac.numerator, frac.denominator


def _resample_signal(signal: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample a (n_samples, n_channels) array and return float32."""
    resampled = resample_poly(signal, up, down, axis=0)
    return resampled.astype(np.float32)


def _write_dat(path: Path, signal: np.ndarray) -> None:
    """Write a (n_samples, n_channels) float32 array as an interleaved binary file."""
    # C-order flatten: ch0_t0, ch1_t0, ..., ch0_t1, ch1_t1, ...
    signal.astype(np.float32, copy=False).flatten(order="C").tofile(path)


def _process_sample(
    sample_id: str,
    group: List[Tuple[np.ndarray, Dict]],
    target_fs: float,
    output_path: Path,
) -> int:
    """
    Process one sample folder: resample all files in *group* and write output.

    Returns the number of files written.
    """
    sample_out_dir = output_path / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate the meta overrides we'll need per sensor type.
    meta_updates: Dict[str, float] = {}
    files_written = 0

    for signal, file_info in group:
        sensor_type = file_info["sensor_type"]
        filename = file_info["filename"]

        if sensor_type == "current":
            source_fs = float(file_info.get("sample_rate_current_hz") or 0.0)
            rate_key = "sample_rate_current_hz"
        else:  # vibration
            source_fs = float(file_info.get("sample_rate_vibro_hz") or 0.0)
            rate_key = "sample_rate_vibro_hz"

        if source_fs <= 0:
            logger.warning(
                "sample %s / %s: source_fs=%s invalid; skipping",
                sample_id, filename, source_fs,
            )
            continue

        if target_fs >= source_fs:
            logger.warning(
                "sample %s / %s: target_fs=%.1f >= source_fs=%.1f; skipping "
                "(downsampler only reduces sampling rate)",
                sample_id, filename, target_fs, source_fs,
            )
            continue

        up, down = _rational_ratio(source_fs, target_fs)
        resampled = _resample_signal(signal, up, down)

        _write_dat(sample_out_dir / filename, resampled)
        meta_updates[rate_key] = target_fs
        files_written += 1

    if files_written == 0:
        # Nothing was written; clean up the created dir if empty.
        try:
            sample_out_dir.rmdir()
        except OSError:
            pass
        return 0

    # Write meta.json: start from the original meta, apply rate updates.
    original_meta = copy.deepcopy(group[0][1].get("meta") or {})
    for key, value in meta_updates.items():
        original_meta[key] = value

    meta_path = sample_out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(original_meta, fh, indent=2)

    return files_written


def resample_dataset(
    data_list: List[np.ndarray],
    metadata_list: List[Dict],
    target_fs: float,
    output_path: Path,
    *,
    overwrite: bool = False,
    max_workers: int = 4,
) -> Dict[str, int]:
    """
    Resample every signal in *data_list* to *target_fs* and reconstruct the
    dataset folder structure on disk.

    Intended to be called directly after ``DataLoader.load_batch()``:

    .. code-block:: python

        loader = DataLoader(Path("data_set_4"))
        preprocessor = ButterworthLPF(cutoff_hz=500)
        data, meta = loader.load_batch(preprocessor=preprocessor)
        stats = resample_dataset(data, meta, target_fs=1000, output_path=Path("data_set_4_1khz"))

    Parameters
    ----------
    data_list : List[np.ndarray]
        Signals as returned by ``DataLoader.load_batch()``.
        Each array has shape ``(n_samples, n_channels)``.
    metadata_list : List[Dict]
        Corresponding file-info dicts (same length as *data_list*).
    target_fs : float
        Desired output sampling rate in Hz. Must be lower than the source
        sampling rate; files that already have ``source_fs <= target_fs`` are
        skipped with a warning.
    output_path : Path | str
        Root directory for the reconstructed dataset.  Sub-folders mirror the
        source sample layout (``output_path/<sample_id>/``).
    overwrite : bool, optional
        If *False* (default) and *output_path* already contains at least one
        ``meta.json`` file, ``FileExistsError`` is raised **before any data is
        written**.  Set to *True* to allow incremental / overwrite runs.
    max_workers : int, optional
        Number of parallel worker threads (one sample per thread).

    Returns
    -------
    dict
        ``{"samples_written": int, "files_written": int, "samples_skipped": int}``

    Raises
    ------
    ValueError
        If *data_list* and *metadata_list* have different lengths, or if
        *target_fs* is not positive.
    FileExistsError
        If *overwrite* is False and output directory already contains data.
    """
    if len(data_list) != len(metadata_list):
        raise ValueError(
            f"data_list and metadata_list must have the same length "
            f"({len(data_list)} vs {len(metadata_list)})"
        )
    if target_fs <= 0:
        raise ValueError(f"target_fs must be positive, got {target_fs}")
    if not data_list:
        logger.warning("resample_dataset called with an empty data_list; nothing to do")
        return {"samples_written": 0, "files_written": 0, "samples_skipped": 0}

    output_path = Path(output_path)

    # Pre-flight: guard against accidental overwrites.
    if not overwrite and output_path.exists():
        existing_metas = list(output_path.rglob("meta.json"))
        if existing_metas:
            raise FileExistsError(
                f"Output path '{output_path}' already contains {len(existing_metas)} "
                f"meta.json file(s). Pass overwrite=True to allow overwriting."
            )

    # Group signals by sample_id.
    groups: Dict[str, List[Tuple[np.ndarray, Dict]]] = {}
    for signal, file_info in zip(data_list, metadata_list):
        sid = file_info["sample_id"]
        groups.setdefault(sid, []).append((signal, file_info))

    total_samples = len(groups)
    logger.info(
        "resample_dataset: %d signal(s) → %d sample folder(s), target_fs=%.1f Hz, "
        "output='%s'",
        len(data_list), total_samples, target_fs, output_path,
    )

    samples_written = 0
    samples_skipped = 0
    total_files_written = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sid = {
            executor.submit(_process_sample, sid, group, target_fs, output_path): sid
            for sid, group in groups.items()
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                n_files = future.result()
                if n_files > 0:
                    samples_written += 1
                    total_files_written += n_files
                else:
                    samples_skipped += 1
            except Exception as exc:
                logger.error("sample %s failed: %s", sid, exc)
                samples_skipped += 1

    logger.info(
        "resample_dataset done: %d/%d samples written, %d file(s) total, "
        "%d sample(s) skipped",
        samples_written, total_samples, total_files_written, samples_skipped,
    )
    return {
        "samples_written": samples_written,
        "files_written": total_files_written,
        "samples_skipped": samples_skipped,
    }
