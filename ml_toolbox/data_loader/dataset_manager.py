"""
Dataset management utilities for motor health monitoring data.
"""
import json
import logging
import numbers
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Set, Tuple, Any, Iterable, cast
from ..data_io import read_current, read_vibro

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manage motor health monitoring dataset."""

    REQUIRED_META_FIELDS = {
        "class": str,
        "electrical_frequency_hz": numbers.Real,
        "load": numbers.Real,
        "sample_rate_current_hz": numbers.Real,
        "sample_rate_vibro_hz": numbers.Real,
        "pwm_frequency_hz": numbers.Real,
    }
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self._index = None
        # Auto-detect binary dtype from folder name suffix.
        # Datasets reconstructed by float_conv.py end with "_float32".
        if self.dataset_path.name.endswith("_float32"):
            self._dat_dtype = np.float32
        else:
            self._dat_dtype = np.float64
    
    def scan_dataset(self) -> Dict:
        """Scan dataset directory and create index based on per-sample meta.json files."""

        dataset_index: Dict[str, Any] = {
            "classes": set(),
            "loads": [],
            "electrical_frequencies_hz": set(),
            "sensor_types": set(),
            "files": [],
        }

        for sample_dir in self._iter_sample_dirs():
            result = cast(
                Tuple[bool, List[str], Optional[Dict[str, Any]]],
                self.validate_sample_dir(sample_dir, return_meta=True),
            )
            ok, errors, meta = result
            if not ok:
                logger.warning("Skipping sample %s: %s", sample_dir.name, "; ".join(errors))
                continue
            if meta is None:
                logger.warning("Skipping sample %s: meta.json parsing produced no metadata", sample_dir.name)
                continue

            class_label = meta.get("class")
            load_value = meta.get("load")
            electrical_freq = meta.get("electrical_frequency_hz")

            if class_label:
                dataset_index["classes"].add(class_label)
            if load_value is not None:
                dataset_index["loads"].append(load_value)
            if electrical_freq is not None:
                dataset_index["electrical_frequencies_hz"].add(electrical_freq)

            found_sensor_file = False
            for file_path in sorted(sample_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.name.lower() == "meta.json":
                    continue
                if file_path.suffix.lower() != ".dat":
                    continue

                sensor_type = self._detect_sensor_type(file_path.name)
                if sensor_type == "unknown":
                    continue

                found_sensor_file = True
                dataset_index["sensor_types"].add(sensor_type)

                file_info = {
                    "path": str(file_path.relative_to(self.dataset_path)),
                    "absolute_path": str(file_path),
                    "sample_id": sample_dir.name,
                    "sample_dir": str(sample_dir.relative_to(self.dataset_path)),
                    "class": class_label,
                    "load": load_value,
                    "electrical_frequency_hz": electrical_freq,
                    "pwm_frequency_hz": meta.get("pwm_frequency_hz"),
                    "sample_rate_current_hz": meta.get("sample_rate_current_hz"),
                    "sample_rate_vibro_hz": meta.get("sample_rate_vibro_hz"),
                    "sensor_type": sensor_type,
                    "filename": file_path.name,
                    "meta": meta,
                }
                dataset_index["files"].append(file_info)

            if not found_sensor_file:
                logger.warning("No sensor data files found in sample %s; skipping", sample_dir.name)

        # Convert sets to sorted lists and deduplicate numeric loads
        dataset_index["classes"] = sorted(list(dataset_index["classes"]))
        dataset_index["sensor_types"] = sorted(list(dataset_index["sensor_types"]))
        dataset_index["electrical_frequencies_hz"] = sorted(list(dataset_index["electrical_frequencies_hz"]))
        dataset_index["loads"] = sorted(list({v for v in dataset_index["loads"] if v is not None}))

        return dataset_index
    
    def _detect_sensor_type(self, filename: str) -> str:
        """Detect sensor type from filename using known channel identifiers."""
        name = filename.lower()
        if "ltr11" in name:
            return "current"
        if "ltr22" in name:
            return "vibration"
        return "unknown"
    
    def get_index(self, force_rescan: bool = False) -> Dict:
        """Get dataset index, optionally forcing a rescan."""
        if self._index is None or force_rescan:
            self._index = self.scan_dataset()
        return self._index
    
    def load_sample(self, file_info: Dict) -> np.ndarray:
        """Load a single data sample."""
        file_path = Path(file_info["absolute_path"])
        sensor_type = file_info["sensor_type"]
        
        if sensor_type == "current":
            return read_current(file_path, dtype=self._dat_dtype)
        elif sensor_type == "vibration":
            return read_vibro(file_path, dtype=self._dat_dtype)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
  
    def filter_files(self,
                     *,
                     classes: Optional[Union[str, Sequence[str]]] = None,
                     loads: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                     frequencies: Optional[Union[float, int, Sequence[Union[float, int]]]] = None,
                     sensor_types: Optional[Union[str, Sequence[str]]] = None,
                     sample_ids: Optional[Union[str, Sequence[str]]] = None) -> List[Dict]:
        """Filter files using strict, exact-match criteria.

        Inputs are validated against index values and invalid entries raise ValueError.
        """
        index = self.get_index()
        filtered = index["files"]

        class_values: Optional[Set[str]] = None
        if classes is not None:
            if isinstance(classes, str):
                class_values = {classes}
            elif isinstance(classes, Sequence):
                class_values = set(classes)
            else:
                raise TypeError("classes must be a string or a sequence of strings")
            if any(not isinstance(value, str) for value in class_values):
                raise TypeError("classes must be a string or a sequence of strings")
            available_classes = set(index["classes"])
            unknown = sorted(class_values - available_classes)
            if unknown:
                raise ValueError(f"Unknown classes: {unknown}. Available classes: {sorted(available_classes)}")

        load_values: Optional[Set[float]] = None
        if loads is not None:
            if isinstance(loads, numbers.Real) and not isinstance(loads, bool):
                load_values = {float(loads)}
            elif isinstance(loads, Sequence):
                load_values = set()
                for value in loads:
                    if not isinstance(value, numbers.Real) or isinstance(value, bool):
                        raise TypeError("loads must be numeric or a sequence of numeric values")
                    load_values.add(float(value))
            else:
                raise TypeError("loads must be numeric or a sequence of numeric values")

            available_loads = {float(v) for v in index["loads"]}
            unknown = sorted(load_values - available_loads)
            if unknown:
                raise ValueError(f"Unknown loads: {unknown}. Available loads: {sorted(available_loads)}")

        frequency_values: Optional[Set[float]] = None
        if frequencies is not None:
            if isinstance(frequencies, numbers.Real) and not isinstance(frequencies, bool):
                frequency_values = {float(frequencies)}
            elif isinstance(frequencies, Sequence):
                frequency_values = set()
                for value in frequencies:
                    if not isinstance(value, numbers.Real) or isinstance(value, bool):
                        raise TypeError("frequencies must be numeric or a sequence of numeric values")
                    frequency_values.add(float(value))
            else:
                raise TypeError("frequencies must be numeric or a sequence of numeric values")

            available_frequencies = {float(v) for v in index["electrical_frequencies_hz"]}
            unknown = sorted(frequency_values - available_frequencies)
            if unknown:
                raise ValueError(
                    f"Unknown frequencies: {unknown}. Available frequencies: {sorted(available_frequencies)}"
                )

        sensor_type_values: Optional[Set[str]] = None
        if sensor_types is not None:
            if isinstance(sensor_types, str):
                sensor_type_values = {sensor_types}
            elif isinstance(sensor_types, Sequence):
                sensor_type_values = set(sensor_types)
            else:
                raise TypeError("sensor_types must be a string or a sequence of strings")
            if any(not isinstance(value, str) for value in sensor_type_values):
                raise TypeError("sensor_types must be a string or a sequence of strings")
            available_sensor_types = set(index["sensor_types"])
            unknown = sorted(sensor_type_values - available_sensor_types)
            if unknown:
                raise ValueError(
                    f"Unknown sensor_types: {unknown}. Available sensor_types: {sorted(available_sensor_types)}"
                )

        sample_id_values: Optional[Set[str]] = None
        if sample_ids is not None:
            if isinstance(sample_ids, str):
                sample_id_values = {sample_ids}
            elif isinstance(sample_ids, Sequence):
                sample_id_values = set(sample_ids)
            else:
                raise TypeError("sample_ids must be a string or a sequence of strings")
            if any(not isinstance(value, str) for value in sample_id_values):
                raise TypeError("sample_ids must be a string or a sequence of strings")
            available_sample_ids = {f.get("sample_id") for f in index["files"]}
            unknown = sorted(sample_id_values - available_sample_ids)
            if unknown:
                raise ValueError(
                    f"Unknown sample_ids: {unknown}. Available sample_ids: {sorted(available_sample_ids)}"
                )

        if class_values is not None:
            filtered = [f for f in filtered if f.get("class") in class_values]
        if load_values is not None:
            filtered = [f for f in filtered if float(f.get("load")) in load_values]
        if frequency_values is not None:
            filtered = [f for f in filtered if float(f.get("electrical_frequency_hz")) in frequency_values]
        if sensor_type_values is not None:
            filtered = [f for f in filtered if f.get("sensor_type") in sensor_type_values]
        if sample_id_values is not None:
            filtered = [f for f in filtered if f.get("sample_id") in sample_id_values]

        return filtered
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        index = self.get_index()
        
        stats: Dict = {
            "total_files": len(index["files"]),
            "classes": len(index["classes"]),
            "loads": len(index["loads"]),
            "electrical_frequencies_hz": len(index["electrical_frequencies_hz"]),
            "sensor_types": len(index["sensor_types"])
        }
        
        # Count files per class
        class_counts = {}
        for cls in index["classes"]:
            class_counts[cls] = len([f for f in index["files"] if f.get("class") == cls])
        stats["files_per_class"] = class_counts
        
        # Count files per sensor type
        sensor_counts = {}
        for sensor_type in index["sensor_types"]:
            sensor_counts[sensor_type] = len([f for f in index["files"] if f.get("sensor_type") == sensor_type])
        stats["files_per_sensor"] = sensor_counts
        
        return stats

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_meta(self, meta: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate meta.json content for required fields and types."""
        errors: List[str] = []

        for field, expected_type in self.REQUIRED_META_FIELDS.items():
            if field not in meta:
                errors.append(f"Missing required field '{field}'")
                continue

            value = meta[field]
            if expected_type is numbers.Real:
                if not isinstance(value, numbers.Real) or isinstance(value, bool):
                    errors.append(f"Field '{field}' must be numeric (int/float)")
            elif not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")

        return len(errors) == 0, errors

    def validate_sample_dir(self, sample_dir: Path, *, return_meta: bool = False, raise_on_error: bool = False) -> Union[Tuple[bool, List[str]], Tuple[bool, List[str], Optional[Dict[str, Any]]]]:
        """Validate a single sample directory (meta.json presence and schema)."""
        sample_dir = Path(sample_dir)
        errors: List[str] = []
        meta: Optional[Dict[str, Any]] = None

        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            errors.append("meta.json not found")
        else:
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Failed to read meta.json: {exc}")
                meta = None

        if meta is not None:
            _, meta_errors = self.validate_meta(meta)
            errors.extend(meta_errors)

        if errors and raise_on_error:
            raise ValueError(f"Validation failed for sample {sample_dir}: {'; '.join(errors)}")

        if return_meta:
            return len(errors) == 0, errors, meta
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_sample_dirs(self) -> Iterable[Path]:
        """Yield candidate sample directories (either under /samples or dataset root)."""
        sample_root = self.dataset_path / "samples"
        if not sample_root.is_dir():
            sample_root = self.dataset_path

        for entry in sorted(sample_root.iterdir()):
            if entry.is_dir():
                yield entry

