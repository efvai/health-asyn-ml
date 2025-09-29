"""
Test categorical features extraction from metadata.
"""

import sys
from pathlib import Path
import numpy as np

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import (
    DataLoader, create_windows_for_ml, 
    FeatureConfig, extract_features_for_ml,
    extract_categorical_features
)


def test_categorical_features():
    """Test extraction of categorical features from metadata."""
    print("Categorical Features Test")
    print("=" * 30)
    
    # 1. Load data with metadata
    print("\n1. Loading data with metadata...")
    data_loader = DataLoader(Path("data_set"))
    
    # Load multiple conditions to get variety in metadata
    healthy_data, healthy_meta = data_loader.load_batch(
        condition="healthy",
        sensor_type="current", 
        max_workers=2
    )
    
    faulty_data, faulty_meta = data_loader.load_batch(
        condition="faulty_bearing",
        sensor_type="current",
        max_workers=2
    )
    
    print(f"✓ Loaded {len(healthy_data)} healthy files")
    print(f"✓ Loaded {len(faulty_data)} faulty bearing files")
    
    # 2. Create windows with metadata
    print("\n2. Creating windows with metadata...")
    
    # Combine data
    all_data = healthy_data + faulty_data
    all_metadata = healthy_meta + faulty_meta
    
    windows, labels, win_metadata = create_windows_for_ml(
        all_data,
        all_metadata,
        window_size=1024,   # Small for quick test
        overlap_ratio=0.5,
        balance_classes=False,
        max_windows_per_class=20  # Limited for testing
    )
    
    print(f"✓ Created {windows.shape[0]} windows")
    print(f"✓ Collected {len(win_metadata)} metadata entries")
    
    # 3. Test categorical feature extraction
    print("\n3. Testing categorical feature extraction...")
    
    categorical_features, categorical_names = extract_categorical_features(win_metadata)
    
    print(f"✓ Categorical features shape: {categorical_features.shape}")
    print(f"✓ Categorical feature names: {len(categorical_names)}")
    
    # 4. Show categorical features details
    print("\n4. Categorical features analysis...")
    
    for i, name in enumerate(categorical_names):
        values = categorical_features[:, i]
        unique_vals = np.unique(values)
        print(f"  {name}:")
        print(f"    Range: {unique_vals}")
        print(f"    Non-zero count: {np.sum(values > 0)}")
    
    # 5. Test combined feature extraction
    print("\n5. Testing combined feature extraction...")
    
    # Extract features with metadata
    all_features, all_names = extract_features_for_ml(
        windows,
        sampling_rate=10000,
        sensor_type="current",
        metadata_list=win_metadata
    )
    
    print(f"✓ Combined features shape: {all_features.shape}")
    print(f"✓ Total feature names: {len(all_names)}")
    
    # 6. Compare with signal-only features
    print("\n6. Comparing with signal-only features...")
    
    signal_features, signal_names = extract_features_for_ml(
        windows,
        sampling_rate=10000,
        sensor_type="current",
        metadata_list=None  # No metadata
    )
    
    print(f"✓ Signal-only features: {signal_features.shape}")
    print(f"✓ Signal-only names: {len(signal_names)}")
    
    categorical_count = len(all_names) - len(signal_names)
    print(f"✓ Added categorical features: {categorical_count}")
    
    # 7. Show example metadata and categorical features
    print("\n7. Example metadata and categorical mapping...")
    
    if win_metadata:
        print("Sample metadata entries:")
        for i in range(min(3, len(win_metadata))):
            meta = win_metadata[i]
            print(f"  Window {i}:")
            print(f"    Condition: {meta.get('condition', 'unknown')}")
            print(f"    Frequency: {meta.get('frequency', 'unknown')}")
            print(f"    Load: {meta.get('load', 'unknown')}")
            print(f"    Categorical features: {categorical_features[i]}")
    
    # 8. Quality check
    print("\n8. Quality check...")
    
    nan_count = np.sum(np.isnan(all_features))
    inf_count = np.sum(np.isinf(all_features))
    
    print(f"  Combined features NaN count: {nan_count}")
    print(f"  Combined features Inf count: {inf_count}")
    
    if nan_count == 0 and inf_count == 0:
        print("  ✅ All features are clean!")
    else:
        print("  ⚠️ Some features have invalid values")
    
    print("\n" + "=" * 30)
    print("✅ Categorical features test completed!")
    print(f"   • Signal features: {len(signal_names)}")
    print(f"   • Categorical features: {categorical_count}")
    print(f"   • Total features: {len(all_names)}")
    print(f"   • Windows processed: {windows.shape[0]}")


if __name__ == "__main__":
    test_categorical_features()