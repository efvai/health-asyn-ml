"""
Quick script to compare files from different conditions
"""

import sys
from pathlib import Path
import numpy as np

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import DatasetManager, DataLoader


def quick_condition_comparison():
    """Compare files from different conditions."""
    
    print("🔍 Quick Condition Comparison")
    print("=" * 60)
    
    dataset_path = Path("data_set")
    data_loader = DataLoader(dataset_path)
    
    conditions = ["healthy", "faulty_bearing", "misalignment"]
    
    for condition in conditions:
        print(f"\n📁 Loading {condition} condition...")
        
        data_list, metadata_list = data_loader.load_batch(
            condition=condition,
            sensor_type="current",
            max_workers=1
        )
        
        if not data_list:
            print(f"   ❌ No files found for {condition}")
            continue
            
        # Analyze first file
        data = data_list[0]
        meta = metadata_list[0]
        
        print(f"   📊 First file: {Path(meta['path']).name}")
        print(f"   📊 Shape: {data.shape}")
        print(f"   📊 Duration: {data.shape[0]/10000:.2f}s")
        
        # Sample a middle portion to avoid startup transients
        start_idx = data.shape[0] // 4
        end_idx = start_idx + 5000  # 0.5 seconds
        sample = data[start_idx:end_idx, :]
        
        for ch in range(2):
            mean_val = np.mean(sample[:, ch])
            std_val = np.std(sample[:, ch])
            min_val = np.min(sample[:, ch])
            max_val = np.max(sample[:, ch])
            rms_val = np.sqrt(np.mean(sample[:, ch]**2))
            
            print(f"   📊 Ch{ch+1}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.2f}, {max_val:.2f}], RMS={rms_val:.3f}")


if __name__ == "__main__":
    quick_condition_comparison()