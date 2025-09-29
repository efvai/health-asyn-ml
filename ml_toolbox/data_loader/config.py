"""
Configuration management for the ML toolbox.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class SensorConfig:
    """Sensor configuration."""
    name: str
    sampling_rate: int
    channels: int
    units: str
    range_min: float
    range_max: float
    description: str


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    version: str
    description: str
    sensors: Dict[str, SensorConfig]
    conditions: List[str]
    loads: List[str]
    frequencies: List[str]
    file_format: str
    created_date: str
    last_updated: str


class ConfigManager:
    """Manage configuration files."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.dataset_config_file = self.config_dir / "dataset_config.json"
        self.sensor_config_file = self.config_dir / "sensor_config.json"
    
    def create_default_config(self) -> DatasetConfig:
        """Create default dataset configuration."""
        
        # Default sensor configurations
        sensors = {
            "current": SensorConfig(
                name="LTR11 Current Sensor",
                sampling_rate=10000,
                channels=2,
                units="Amperes",
                range_min=-50.0,
                range_max=50.0,
                description="Current measurement from induction motor"
            ),
            "vibration": SensorConfig(
                name="LTR22 Vibration Sensor", 
                sampling_rate=26041,
                channels=4,
                units="m/s²",
                range_min=-100.0,
                range_max=100.0,
                description="Vibration measurement from induction motor"
            )
        }
        
        config = DatasetConfig(
            name="Motor Health Monitoring Dataset",
            version="1.0.0",
            description="Induction motor health monitoring data with current and vibration sensors",
            sensors=sensors,
            conditions=["healthy", "faulty_bearing", "misalignment", "system_misalignment"],
            loads=["no_load", "under_load"],
            frequencies=["10hz", "20hz", "30hz", "40hz"],
            file_format="binary_double",
            created_date="2025-09-21",
            last_updated="2025-09-21"
        )
        
        return config
    
    def save_config(self, config: DatasetConfig):
        """Save configuration to JSON file."""
        # Convert dataclass to dict with special handling for nested dataclasses
        config_dict = asdict(config)
        
        with open(self.dataset_config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self) -> DatasetConfig:
        """Load configuration from JSON file."""
        if not self.dataset_config_file.exists():
            # Create and save default config if none exists
            default_config = self.create_default_config()
            self.save_config(default_config)
            return default_config
        
        with open(self.dataset_config_file, 'r') as f:
            data = json.load(f)
        
        # Convert sensor configs back to dataclasses
        sensors = {}
        for name, sensor_data in data['sensors'].items():
            sensors[name] = SensorConfig(**sensor_data)
        
        data['sensors'] = sensors
        return DatasetConfig(**data)
    
    def update_config(self, **kwargs):
        """Update specific configuration fields."""
        config = self.load_config()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Update last_updated timestamp
        from datetime import datetime
        config.last_updated = datetime.now().strftime("%Y-%m-%d")
        
        self.save_config(config)
    
    def add_sensor(self, sensor_type: str, sensor_config: SensorConfig):
        """Add a new sensor configuration."""
        config = self.load_config()
        config.sensors[sensor_type] = sensor_config
        self.save_config(config)
    
    def get_sensor_config(self, sensor_type: str) -> Optional[SensorConfig]:
        """Get configuration for a specific sensor."""
        config = self.load_config()
        return config.sensors.get(sensor_type)
    
    def export_config(self, output_file: Path):
        """Export configuration to a different file."""
        config = self.load_config()
        with open(output_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def import_config(self, input_file: Path):
        """Import configuration from a file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Convert sensor configs
        sensors = {}
        for name, sensor_data in data['sensors'].items():
            sensors[name] = SensorConfig(**sensor_data)
        
        data['sensors'] = sensors
        config = DatasetConfig(**data)
        self.save_config(config)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        try:
            config = self.load_config()
            
            # Check required fields
            if not config.name:
                issues.append("Dataset name is required")
            
            if not config.sensors:
                issues.append("At least one sensor configuration is required")
            
            # Validate sensor configs
            for sensor_type, sensor_config in config.sensors.items():
                if sensor_config.sampling_rate <= 0:
                    issues.append(f"Sensor {sensor_type}: sampling rate must be positive")
                
                if sensor_config.channels <= 0:
                    issues.append(f"Sensor {sensor_type}: number of channels must be positive")
                
                if sensor_config.range_min >= sensor_config.range_max:
                    issues.append(f"Sensor {sensor_type}: range_min must be less than range_max")
            
            # Check conditions
            if not config.conditions:
                issues.append("At least one condition must be defined")
            
            # Check loads
            if not config.loads:
                issues.append("At least one load condition must be defined")
                
        except Exception as e:
            issues.append(f"Configuration file error: {str(e)}")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        try:
            config = self.load_config()
            
            summary = {
                "name": config.name,
                "version": config.version,
                "sensors": list(config.sensors.keys()),
                "sensor_details": {
                    name: {
                        "channels": sensor.channels,
                        "sampling_rate": sensor.sampling_rate,
                        "units": sensor.units
                    }
                    for name, sensor in config.sensors.items()
                },
                "conditions": config.conditions,
                "loads": config.loads,
                "frequencies": len(config.frequencies),
                "file_format": config.file_format,
                "last_updated": config.last_updated
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to load configuration: {str(e)}"}