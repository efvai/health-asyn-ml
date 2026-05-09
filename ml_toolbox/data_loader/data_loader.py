"""
Efficient data loading pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from .dataset_manager import DatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logger.propagate = False


class DataLoader:
    """Efficient data loader with concurrent processing."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_manager = DatasetManager(dataset_path)
        self._index = None
    
    @property
    def index(self):
        """Lazy load dataset index."""
        if self._index is None:
            self._index = self.dataset_manager.get_index()
        return self._index
    
    def load_batch(self,
                   max_workers: int = 4,
                   *,
                   classes: Optional[Union[str, Sequence[str]]] = None,
                   loads: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                   frequencies: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                   sensor_types: Optional[Union[str, Sequence[str]]] = None,
                   sample_ids: Optional[Union[str, Sequence[str]]] = None,
                   preprocessor=None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load batch of data with optional filtering.
        
        Args:
            max_workers: Number of parallel workers for loading.
            classes: One or more class labels to include.
            loads: One or more numeric load values to include.
            frequencies: One or more electrical frequencies (Hz) to include.
            sensor_types: One or more sensor types to include.
            sample_ids: Optional sample directory names to include (e.g., '0001').
            preprocessor: Optional object with an ``apply(signal, fs)`` method
                (e.g. ``ButterworthLPF``).  When provided, each loaded array is
                passed through ``preprocessor.apply(data, fs)`` before being
                returned.  The sampling rate ``fs`` is resolved automatically
                from per-sample metadata (``sample_rate_current_hz`` /
                ``sample_rate_vibro_hz``).
            
        Returns:
            Tuple of (data_list, metadata_list)
        """
        
        # Filter files based on criteria
        filtered_files = self.dataset_manager.filter_files(
            classes=classes,
            loads=loads,
            frequencies=frequencies,
            sensor_types=sensor_types,
            sample_ids=sample_ids,
        )
        
        if not filtered_files:
            logger.warning(
                "No files found matching criteria: "
                f"classes={classes}, "
                f"loads={loads}, "
                f"frequencies={frequencies}, "
                f"sensor_types={sensor_types}"
            )
            return [], []
        
        logger.info(f"Loading {len(filtered_files)} files with {max_workers} workers")
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data_futures = [
                    executor.submit(self.dataset_manager.load_sample, file_info) 
                    for file_info in filtered_files
            ]
            
            data_list = []
            metadata_list = []
            
            for future, file_info in zip(data_futures, filtered_files):
                try:
                    data = future.result()
                    if preprocessor is not None:
                        sensor_type = file_info.get('sensor_type', '')
                        if sensor_type == 'vibration':
                            fs = file_info['sample_rate_vibro_hz']
                        else:
                            fs = file_info['sample_rate_current_hz']
                        data = preprocessor.apply(data, fs)
                    data_list.append(data)
                    metadata_list.append(file_info)
                except Exception as e:
                    logger.error(f"Error loading {file_info['path']}: {e}")
        
        logger.info(f"Successfully loaded {len(data_list)} files")
        return data_list, metadata_list


def extract_features_lazy(
    dataset_path: Path,
    file_list: List[Dict],
    preprocessor,
    window_size: int,
    overlap_ratio: float,
    shuffle: bool,
    random_state: Optional[int],
    sensor_type: str,
    feature_config,
    class_to_int: Optional[Dict[str, int]] = None,
    progress_callback=None,
    dataset_manager=None,
) -> Tuple[np.ndarray, np.ndarray, List, List[Dict], Dict]:
    """
    Extract features lazily — one file at a time — to minimise peak RAM.

    For every file: load signal → apply preprocessor → slice windows →
    extract features → free signal and windows immediately.
    Only feature vectors (far smaller than raw signals) accumulate in RAM.

    Parameters
    ----------
    dataset_path : Path
        Root path of the dataset.
    file_list : List[Dict]
        Pre-filtered file-info dicts from DatasetManager.filter_files() or
        get_filtered_file_list().
    preprocessor :
        Object with ``.apply(signal, fs)`` method (e.g. ButterworthLPF), or None.
    window_size : int
        Number of samples per window.
    overlap_ratio : float
        Fraction of overlap between successive windows (0 – <1).
    shuffle : bool
        Whether to shuffle windows after extraction.
    random_state : Optional[int]
        Seed for the shuffle RNG.
    sensor_type : str
        ``"vibration"`` or ``"current"``.
    feature_config :
        ``FeatureConfig`` instance (will be copied & sensor-profile applied).
        Pass None to use the default sensor profile.
    class_to_int : Optional[Dict[str, int]]
        Mapping ``{class_name: label_int}``.  Built automatically from
        *file_list* when None (sorted → 0-indexed).
    progress_callback : callable(done: int, total: int) | None
        Called after each file with ``(files_done, total_files)``.

    Returns
    -------
    features : np.ndarray, shape (n_windows, n_features), float32
    labels   : np.ndarray, shape (n_windows,), int32
    feature_names : List[str]
    win_metadata  : List[Dict]  — one dict per window
    label_map     : Dict[int, str]  — reverse of class_to_int
    """
    from .feature_extraction import FeatureExtractor
    from .features import FeatureConfig

    if not file_list:
        raise ValueError("file_list is empty — check your filter criteria")

    # ── Build class encoding ──────────────────────────────────────────────────
    if class_to_int is None:
        unique_classes = sorted(set(f["class"] for f in file_list))
        class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    label_map: Dict[int, str] = {v: k for k, v in class_to_int.items()}

    # ── Prepare feature config ────────────────────────────────────────────────
    if feature_config is None:
        feature_config = FeatureConfig.for_sensor(sensor_type)
    else:
        feature_config = feature_config.copy()
        feature_config.apply_sensor_profile(sensor_type, override=False)

    step_size = max(1, int(window_size * (1 - overlap_ratio)))
    dm = dataset_manager if dataset_manager is not None else DatasetManager(dataset_path)
    extractor = FeatureExtractor(feature_config)

    all_feature_chunks: List[np.ndarray] = []
    all_labels: List[int] = []
    all_meta: List[Dict] = []
    feature_names: Optional[List] = None

    total = len(file_list)
    for i, file_info in enumerate(file_list):
        try:
            # ── Resolve sampling rate ─────────────────────────────────────
            if sensor_type == "current":
                fs = float(file_info.get("sample_rate_current_hz") or 1.0)
            else:
                fs = float(file_info.get("sample_rate_vibro_hz") or 1.0)

            # ── Load signal ───────────────────────────────────────────────
            signal = dm.load_sample(file_info)   # (n_samples, n_channels) float64
            if signal.ndim == 1:
                signal = signal[:, np.newaxis]
            n_samples, n_channels = signal.shape

            # ── Apply preprocessor ────────────────────────────────────────
            if preprocessor is not None:
                signal = preprocessor.apply(signal.astype(np.float64), fs=fs)
            signal = signal.astype(np.float32)

            # ── Skip files shorter than one window ────────────────────────
            if n_samples < window_size:
                logger.warning(
                    "File %s: %d samples < window_size %d; skipping",
                    file_info.get("path", "?"), n_samples, window_size,
                )
                del signal
                if progress_callback is not None:
                    progress_callback(i + 1, total)
                continue

            # ── Slice into windows ────────────────────────────────────────
            starts = range(0, n_samples - window_size + 1, step_size)
            windows_arr = np.stack(
                [signal[s: s + window_size] for s in starts], axis=0
            )   # (n_win, window_size, n_channels)
            del signal

            n_win = windows_arr.shape[0]
            channel_names = [f"ch{j + 1}" for j in range(n_channels)]
            sampling_rates_list = [fs] * n_win

            # ── Extract features ──────────────────────────────────────────
            feats, f_names = extractor.extract_features_batch(
                windows_arr,
                channel_names=channel_names,
                sampling_rates=sampling_rates_list,
            )
            del windows_arr

            if feature_names is None:
                feature_names = f_names

            # ── Guard: skip unknown classes ───────────────────────────────
            cls_str = file_info.get("class")
            if cls_str not in class_to_int:
                logger.warning(
                    "File %s: class %r not in class_to_int; skipping",
                    file_info.get("path", "?"), cls_str,
                )
                if progress_callback is not None:
                    progress_callback(i + 1, total)
                continue

            label_int = class_to_int[cls_str]
            base_meta = {
                "sample_id": file_info.get("sample_id"),
                "class": cls_str,
                "load": file_info.get("load"),
                "electrical_frequency_hz": file_info.get("electrical_frequency_hz"),
                "sensor_type": file_info.get("sensor_type"),
                "sample_rate_vibro_hz": file_info.get("sample_rate_vibro_hz"),
                "sample_rate_current_hz": file_info.get("sample_rate_current_hz"),
            }
            for w_idx, start in enumerate(starts):
                all_meta.append({**base_meta, "window_id": w_idx, "start_sample": start})

            all_feature_chunks.append(feats.astype(np.float32))
            all_labels.extend([label_int] * n_win)

        except Exception as e:
            logger.error("Error processing %s: %s", file_info.get("path", "?"), e)

        if progress_callback is not None:
            progress_callback(i + 1, total)

    if not all_feature_chunks:
        raise RuntimeError("No features extracted — all files failed or produced no windows")

    features = np.concatenate(all_feature_chunks, axis=0)
    labels = np.array(all_labels, dtype=np.int32)

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(labels))
        features = features[idx]
        labels = labels[idx]
        all_meta = [all_meta[j] for j in idx]

    return features, labels, feature_names or [], all_meta, label_map
