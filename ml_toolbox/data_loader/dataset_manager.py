"""
Dataset management utilities for motor health monitoring data.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from ..data_io import read_current, read_vibro

class DatasetManager:
    """Manage motor health monitoring dataset."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self._index = None
    
    def scan_dataset(self) -> Dict:
        """Scan dataset directory and create index."""
        dataset_index = {
            "conditions": set(),
            "loads": set(), 
            "frequencies": set(),
            "sensor_types": set(),
            "files": []
        }
        
        # Scan the data_set directory
        for condition_dir in self.dataset_path.iterdir():
            if condition_dir.is_dir() and condition_dir.name != "metadata":
                condition = condition_dir.name.replace(" ", "_")  # Normalize naming
                dataset_index["conditions"].add(condition)
                
                for load_dir in condition_dir.iterdir():
                    if load_dir.is_dir():
                        load = load_dir.name.replace(" ", "_")  # Normalize naming
                        dataset_index["loads"].add(load)
                        
                        for freq_dir in load_dir.iterdir():
                            if freq_dir.is_dir():
                                freq_raw = freq_dir.name.replace(" ", "_")  # Normalize naming
                                # Extract base frequency (remove experiment number)
                                freq = self._extract_base_frequency(freq_raw)
                                dataset_index["frequencies"].add(freq)
                                
                                # Scan for .dat files
                                for file_path in freq_dir.glob("*.dat"):
                                    sensor_type = self._detect_sensor_type(file_path.name)
                                    dataset_index["sensor_types"].add(sensor_type)
                                    
                                    file_info = {
                                        "path": str(file_path.relative_to(self.dataset_path)),
                                        "absolute_path": str(file_path),
                                        "condition": condition,
                                        "load": load,
                                        "frequency": freq,  # Base frequency (e.g., "10hz")
                                        "frequency_dir": freq_raw,  # Full directory name (e.g., "10hz_1")
                                        "sensor_type": sensor_type,
                                        "filename": file_path.name
                                    }
                                    dataset_index["files"].append(file_info)
        
        # Convert sets to sorted lists for JSON serialization
        for key in ["conditions", "loads", "frequencies", "sensor_types"]:
            dataset_index[key] = sorted(list(dataset_index[key]))
        
        return dataset_index
    
    def _detect_sensor_type(self, filename: str) -> str:
        """Detect sensor type from filename."""
        if "LTR11" in filename:
            return "current"
        elif "LTR22" in filename:
            return "vibration"
        else:
            return "unknown"
    
    def _extract_base_frequency(self, freq_dir_name: str) -> str:
        """Extract base frequency from directory name (remove experiment number)."""
        # Examples: "10hz_1" -> "10hz", "20hz_2" -> "20hz", "30hz" -> "30hz"
        import re
        # Match pattern like "10hz", "20hz", etc. (before any underscore)
        match = re.match(r'^(\d+hz)', freq_dir_name)
        if match:
            return match.group(1)
        else:
            # If no pattern matches, return the original (for backward compatibility)
            return freq_dir_name
    
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
            return read_current(file_path)
        elif sensor_type == "vibration":
            return read_vibro(file_path)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
  
    def filter_files(self, 
                     condition: Optional[str] = None,
                     load: Optional[str] = None,
                     frequency: Optional[str] = None,
                     sensor_type: Optional[str] = None) -> List[Dict]:
        """Filter files based on criteria."""
        index = self.get_index()
        filtered = index["files"]
        
        if condition:
            condition = condition.replace(" ", "_")
            filtered = [f for f in filtered if f["condition"] == condition]
        if load:
            load = load.replace(" ", "_")
            filtered = [f for f in filtered if f["load"] == load]
        if frequency:
            frequency = frequency.replace(" ", "_")
            filtered = [f for f in filtered if f["frequency"] == frequency]
        if sensor_type:
            filtered = [f for f in filtered if f["sensor_type"] == sensor_type]
            
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        index = self.get_index()
        
        stats: Dict = {
            "total_files": len(index["files"]),
            "conditions": len(index["conditions"]),
            "loads": len(index["loads"]),
            "frequencies": len(index["frequencies"]),
            "sensor_types": len(index["sensor_types"])
        }
        
        # Count files per condition
        condition_counts = {}
        for condition in index["conditions"]:
            condition_counts[condition] = len([f for f in index["files"] if f["condition"] == condition])
        stats["files_per_condition"] = condition_counts
        
        # Count files per sensor type
        sensor_counts = {}
        for sensor_type in index["sensor_types"]:
            sensor_counts[sensor_type] = len([f for f in index["files"] if f["sensor_type"] == sensor_type])
        stats["files_per_sensor"] = sensor_counts
        
        return stats