"""
Debug script to check raw vs processed data and compare with MATLAB expectations
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox.data_io.read_raw import read_raw
from ml_toolbox.data_io.read_current import read_current


def debug_data_processing():
    """Debug the data processing pipeline step by step."""
    
    print("🔧 Debug Data Processing Pipeline")
    print("=" * 60)
    
    # Find a specific file to analyze
    dataset_path = Path("data_set")
    healthy_path = dataset_path / "healthy" / "no load" / "10hz 1"
    
    # Look for .dat files
    dat_files = list(healthy_path.glob("*.dat"))
    if not dat_files:
        print("❌ No .dat files found")
        return
    
    file_path = dat_files[0]
    print(f"📁 Analyzing file: {file_path}")
    
    # Step 1: Read raw data
    print("\n🔍 Step 1: Reading raw data...")
    raw_data = read_raw(file_path, 2)
    print(f"   Raw data shape: {raw_data.shape}")
    print(f"   Raw data range: [{np.min(raw_data):.6f}, {np.max(raw_data):.6f}]")
    print(f"   Raw data mean: [{np.mean(raw_data[:, 0]):.6f}, {np.mean(raw_data[:, 1]):.6f}]")
    print(f"   Raw data std: [{np.std(raw_data[:, 0]):.6f}, {np.std(raw_data[:, 1]):.6f}]")
    
    # Step 2: Process with read_current
    print("\n🔍 Step 2: Processing with read_current...")
    processed_data = read_current(file_path)
    print(f"   Processed data shape: {processed_data.shape}")
    print(f"   Processed data range: [{np.min(processed_data):.6f}, {np.max(processed_data):.6f}]")
    print(f"   Processed data mean: [{np.mean(processed_data[:, 0]):.6f}, {np.mean(processed_data[:, 1]):.6f}]")
    print(f"   Processed data std: [{np.std(processed_data[:, 0]):.6f}, {np.std(processed_data[:, 1]):.6f}]")
    
    # Step 3: Manual processing step-by-step (following MATLAB)
    print("\n🔍 Step 3: Manual step-by-step processing...")
    
    # 3a. Remove mean (DC offset)
    data_dc_removed = raw_data - np.mean(raw_data, axis=0)
    print(f"   After DC removal - Mean: [{np.mean(data_dc_removed[:, 0]):.6f}, {np.mean(data_dc_removed[:, 1]):.6f}]")
    
    # 3b. Median filtering
    from scipy.signal import medfilt
    signal_medfilt = np.zeros_like(data_dc_removed)
    for channel in range(data_dc_removed.shape[1]):
        signal_medfilt[:, channel] = medfilt(data_dc_removed[:, channel], kernel_size=21)
    
    # 3c. Calculate residual
    residual = data_dc_removed - signal_medfilt
    print(f"   Residual range: [{np.min(residual):.6f}, {np.max(residual):.6f}]")
    
    # 3d. MAD calculation (MATLAB style - single value)
    residual_flat = residual.flatten()  # Flatten to match MATLAB behavior
    mad_val_matlab_style = np.median(np.abs(residual_flat - np.median(residual_flat)))
    print(f"   MAD (MATLAB style): {mad_val_matlab_style:.6f}")
    
    # 3e. MAD calculation (Python style - per channel)
    mad_val_python_style = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    print(f"   MAD (Python style): [{mad_val_python_style[0]:.6f}, {mad_val_python_style[1]:.6f}]")
    
    # Create comparison plot
    print("\n📊 Creating comparison plots...")
    with PdfPages("debug_processing.pdf") as pdf:
        
        # Plot 1: Raw vs Processed comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample a window for display
        start_idx = 5000
        end_idx = start_idx + 2000  # 0.2 seconds
        time_axis = np.arange(end_idx - start_idx) / 10000 * 1000  # Convert to ms
        
        # Raw data
        axes[0, 0].plot(time_axis, raw_data[start_idx:end_idx, 0], 'b-', linewidth=0.8)
        axes[0, 0].set_title('Raw Data - Channel 1')
        axes[0, 0].set_ylabel('ADC Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_axis, raw_data[start_idx:end_idx, 1], 'r-', linewidth=0.8)
        axes[0, 1].set_title('Raw Data - Channel 2')
        axes[0, 1].set_ylabel('ADC Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processed data
        axes[1, 0].plot(time_axis, processed_data[start_idx:end_idx, 0], 'b-', linewidth=0.8)
        axes[1, 0].set_title('Processed Data - Channel 1')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('ADC Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_axis, processed_data[start_idx:end_idx, 1], 'r-', linewidth=0.8)
        axes[1, 1].set_title('Processed Data - Channel 2')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('ADC Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 2: Processing steps
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # DC removed
        axes[0, 0].plot(time_axis, data_dc_removed[start_idx:end_idx, 0], 'g-', linewidth=0.8)
        axes[0, 0].set_title('After DC Removal - Channel 1')
        axes[0, 0].set_ylabel('ADC Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_axis, data_dc_removed[start_idx:end_idx, 1], 'g-', linewidth=0.8)
        axes[0, 1].set_title('After DC Removal - Channel 2')
        axes[0, 1].set_ylabel('ADC Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Median filtered
        axes[1, 0].plot(time_axis, signal_medfilt[start_idx:end_idx, 0], 'm-', linewidth=0.8)
        axes[1, 0].set_title('Median Filtered - Channel 1')
        axes[1, 0].set_ylabel('ADC Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_axis, signal_medfilt[start_idx:end_idx, 1], 'm-', linewidth=0.8)
        axes[1, 1].set_title('Median Filtered - Channel 2')
        axes[1, 1].set_ylabel('ADC Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Residual
        axes[2, 0].plot(time_axis, residual[start_idx:end_idx, 0], 'orange', linewidth=0.8)
        axes[2, 0].set_title('Residual - Channel 1')
        axes[2, 0].set_xlabel('Time (ms)')
        axes[2, 0].set_ylabel('ADC Value')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time_axis, residual[start_idx:end_idx, 1], 'orange', linewidth=0.8)
        axes[2, 1].set_title('Residual - Channel 2')
        axes[2, 1].set_xlabel('Time (ms)')
        axes[2, 1].set_ylabel('ADC Value')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("✅ Debug analysis complete! Check 'debug_processing.pdf'")
    
    # Check if data looks reasonable
    if np.abs(np.mean(processed_data)) < 0.001:  # Mean should be near zero
        print("✅ DC offset removal appears correct")
    else:
        print("⚠️  DC offset may not be fully removed")
    
    if np.std(processed_data) > 0.1:  # Should have some variation
        print("✅ Signal has reasonable variation")
    else:
        print("⚠️  Signal variation seems too low")


if __name__ == "__main__":
    debug_data_processing()