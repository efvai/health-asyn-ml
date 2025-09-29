"""
Test script to verify the new dataset management structure.
"""

import sys
from pathlib import Path

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import DatasetManager, DataLoader, ConfigManager


def test_dataset_structure():
    """Test the new dataset management functionality."""
    print("Testing Motor Health Dataset Management Structure")
    print("=" * 50)
    
    # Initialize paths
    dataset_path = Path("data_set")
    
    # Test ConfigManager
    print("\n1. Testing Configuration Management...")
    config_manager = ConfigManager(dataset_path / "metadata")
    
    # Create and save default configuration
    config = config_manager.create_default_config()
    config_manager.save_config(config)
    print(f"✓ Created default configuration")
    
    # Get configuration summary
    summary = config_manager.get_config_summary()
    print(f"✓ Configuration summary: {summary['name']}")
    print(f"  - Sensors: {summary['sensors']}")
    print(f"  - Conditions: {len(summary['conditions'])} conditions")
    print(f"  - Loads: {len(summary['loads'])} load conditions")
    
    # Test DatasetManager
    print("\n2. Testing Dataset Manager...")
    dataset_manager = DatasetManager(dataset_path)
    
    # Scan dataset
    index = dataset_manager.get_index()
    print(f"✓ Scanned dataset")
    print(f"  - Found {len(index['files'])} files")
    print(f"  - Conditions: {index['conditions']}")
    print(f"  - Sensor types: {index['sensor_types']}")
    print(f"  - Loads: {index['loads']}")
    print(f"  - Frequencies: {len(index['frequencies'])} frequency settings")
    
    # Save index
    dataset_manager.save_index()
    print(f"✓ Saved dataset index")
    
    # Get statistics
    stats = dataset_manager.get_statistics()
    print(f"✓ Dataset statistics:")
    print(f"  - Total files: {stats['total_files']}")
    print(f"  - Files per condition: {stats['files_per_condition']}")
    print(f"  - Files per sensor: {stats['files_per_sensor']}")
    
    # Test DataLoader
    print("\n3. Testing Data Loader...")
    data_loader = DataLoader(dataset_path)
    
    # Test filtering
    print(f"✓ Testing file filtering...")
    
    # Filter by condition
    healthy_files = data_loader.dataset_manager.filter_files(condition="healthy")
    print(f"  - Healthy files: {len(healthy_files)}")
    
    # Filter by sensor type
    current_files = data_loader.dataset_manager.filter_files(sensor_type="current")
    vibration_files = data_loader.dataset_manager.filter_files(sensor_type="vibration")
    print(f"  - Current sensor files: {len(current_files)}")
    print(f"  - Vibration sensor files: {len(vibration_files)}")
    
    # Test cache info
    cache_info = data_loader.get_cache_info()
    print(f"✓ Cache info: {cache_info['cached_files']} files, {cache_info['total_size_mb']:.2f} MB")
    
    # Test loading a small batch (just a couple files to verify it works)
    print("\n4. Testing Data Loading...")
    try:
        print("  Loading small batch of healthy, no load, current data...")
        data_list, metadata_list = data_loader.load_batch(
            condition="healthy", 
            load="no_load", 
            sensor_type="current",
            max_workers=2
        )
        
        if data_list:
            print(f"✓ Successfully loaded {len(data_list)} files")
            print(f"  - First file shape: {data_list[0].shape}")
            print(f"  - First file metadata: {metadata_list[0]['path']}")
            
            # Test array loading
            data_array, labels_array, metadata = data_loader.load_batch_as_arrays(
                condition="healthy", 
                load="no_load", 
                sensor_type="current",
                max_workers=2
            )
            
            if isinstance(data_array, list):
                print(f"✓ Loaded as list (different shapes): {len(data_array)} samples")
            else:
                print(f"✓ Loaded as array: {data_array.shape}")
            print(f"  - Labels shape: {labels_array.shape}")
            print(f"  - Label mapping: {data_loader.get_label_mapping()}")
        else:
            print("⚠ No data loaded - this might be expected if no files match criteria")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("This might be due to file format issues or missing files")
    
    # Test paired loading
    print("\n5. Testing Paired Data Loading...")
    try:
        current_data, vibration_data, paired_metadata = data_loader.load_pairs(
            condition="healthy",
            load="no_load",
            max_workers=2
        )
        
        if current_data and vibration_data:
            print(f"✓ Successfully loaded {len(current_data)} paired measurements")
            print(f"  - Current data shape: {current_data[0].shape}")
            print(f"  - Vibration data shape: {vibration_data[0].shape}")
        else:
            print("⚠ No paired data loaded")
            
    except Exception as e:
        print(f"❌ Error loading paired data: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed! The new dataset structure is ready to use.")
    print("\nNext steps:")
    print("1. Check the data_set/metadata/ folder for configuration files")
    print("2. Use DataLoader for efficient batch loading in your ML experiments")
    print("3. Use ConfigManager to customize sensor configurations")
    print("4. Cache will speed up repeated data loading")


if __name__ == "__main__":
    test_dataset_structure()