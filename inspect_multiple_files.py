"""
Script to inspect multiple current signal files and compare their characteristics.
This will help identify if there are consistent differences between files.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import DatasetManager, DataLoader


def inspect_multiple_files(output_pdf="file_comparison.pdf", 
                          condition="healthy", 
                          max_files=3,
                          window_duration=0.5,
                          sampling_rate=10000):
    """
    Load and inspect multiple current signal files for comparison.
    
    Args:
        output_pdf: Output PDF filename
        condition: Condition to load (healthy, faulty_bearing, etc.)
        max_files: Maximum number of files to compare
        window_duration: Duration of sample window to show (in seconds)
        sampling_rate: Sampling rate in Hz
    """
    
    print(f"🔍 Inspecting multiple files for condition: {condition}")
    print("=" * 60)
    
    # Initialize data loader
    dataset_path = Path("data_set")
    data_loader = DataLoader(dataset_path)
    
    # Load multiple current signal files
    data_list, metadata_list = data_loader.load_batch(
        condition=condition,
        sensor_type="current",
        max_workers=1
    )
    
    if not data_list:
        print(f"❌ No current signal files found for condition: {condition}")
        return
    
    # Limit to max_files
    num_files = min(max_files, len(data_list))
    data_list = data_list[:num_files]
    metadata_list = metadata_list[:num_files]
    
    print(f"📁 Found {len(data_list)} files, analyzing first {num_files}")
    
    # Calculate window parameters
    window_samples = int(window_duration * sampling_rate)
    
    # Create PDF with comparisons
    with PdfPages(output_pdf) as pdf:
        
        # Page 1: File overview
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        overview_text = f"FILE COMPARISON ANALYSIS\n{'=' * 50}\n\n"
        
        for i, (data, meta) in enumerate(zip(data_list, metadata_list)):
            overview_text += f"File {i+1}:\n"
            overview_text += f"  Path: {meta['path']}\n"
            overview_text += f"  Shape: {data.shape} (samples, channels)\n"
            overview_text += f"  Duration: {data.shape[0] / sampling_rate:.2f}s\n"
            overview_text += f"  Mean: [{np.mean(data[:, 0]):.2f}, {np.mean(data[:, 1]):.2f}]\n"
            overview_text += f"  Std: [{np.std(data[:, 0]):.2f}, {np.std(data[:, 1]):.2f}]\n"
            overview_text += f"  Min: [{np.min(data[:, 0]):.2f}, {np.min(data[:, 1]):.2f}]\n"
            overview_text += f"  Max: [{np.max(data[:, 0]):.2f}, {np.max(data[:, 1]):.2f}]\n\n"
        
        ax.text(0.05, 0.95, overview_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.title('File Overview', fontsize=16, fontweight='bold')
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Signal comparison - first window from each file
        fig, axes = plt.subplots(num_files, 2, figsize=(15, 4*num_files), sharex=True)
        if num_files == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for file_idx, (data, meta) in enumerate(zip(data_list, metadata_list)):
            # Extract first window
            start_sample = 1000  # Start at 0.1s to avoid startup transients
            end_sample = start_sample + window_samples
            window = data[start_sample:end_sample, :]
            
            time_axis = np.linspace(0, window_duration, window_samples) * 1000  # Convert to ms
            
            # Plot each channel
            for ch in range(2):
                ax = axes[file_idx, ch]
                ax.plot(time_axis, window[:, ch], color=colors[file_idx % len(colors)], linewidth=0.8)
                ax.set_ylabel(f'File {file_idx+1}\nCh{ch+1} (ADC)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, window_duration * 1000)
                
                # Add statistics
                mean_val = np.mean(window[:, ch])
                std_val = np.std(window[:, ch])
                ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                if file_idx == 0:
                    ax.set_title(f'Channel {ch+1}', fontsize=12, fontweight='bold')
        
        # Set common x-label
        for ch in range(2):
            axes[-1, ch].set_xlabel('Time (ms)', fontsize=12)
        
        plt.suptitle(f'Signal Comparison - First {window_duration}s Window\n(Starting at 0.1s)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Overlay comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for file_idx, (data, meta) in enumerate(zip(data_list, metadata_list)):
            # Extract same window as before
            start_sample = 1000
            end_sample = start_sample + window_samples
            window = data[start_sample:end_sample, :]
            
            time_axis = np.linspace(0, window_duration, window_samples) * 1000
            
            # Plot both channels overlaid
            for ch in range(2):
                ax = axes[ch]
                ax.plot(time_axis, window[:, ch], 
                       color=colors[file_idx % len(colors)], 
                       linewidth=0.8, 
                       label=f'File {file_idx+1}: {Path(meta["path"]).name}',
                       alpha=0.8)
                ax.set_xlabel('Time (ms)', fontsize=12)
                ax.set_ylabel(f'Channel {ch+1} (ADC units)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                ax.set_title(f'Channel {ch+1} - All Files Overlaid', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Statistical comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect statistics
        file_stats = []
        for data, meta in zip(data_list, metadata_list):
            stats = {
                'name': Path(meta['path']).name,
                'ch1_mean': np.mean(data[:, 0]),
                'ch1_std': np.std(data[:, 0]),
                'ch2_mean': np.mean(data[:, 1]),
                'ch2_std': np.std(data[:, 1]),
                'ch1_range': np.max(data[:, 0]) - np.min(data[:, 0]),
                'ch2_range': np.max(data[:, 1]) - np.min(data[:, 1])
            }
            file_stats.append(stats)
        
        # Plot comparisons
        file_names = [s['name'] for s in file_stats]
        
        # Mean comparison
        ch1_means = [s['ch1_mean'] for s in file_stats]
        ch2_means = [s['ch2_mean'] for s in file_stats]
        x_pos = np.arange(len(file_names))
        
        axes[0,0].bar(x_pos - 0.2, ch1_means, 0.4, label='Channel 1', alpha=0.7)
        axes[0,0].bar(x_pos + 0.2, ch2_means, 0.4, label='Channel 2', alpha=0.7)
        axes[0,0].set_xlabel('Files')
        axes[0,0].set_ylabel('Mean (ADC units)')
        axes[0,0].set_title('Mean Values Comparison')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([f'F{i+1}' for i in range(len(file_names))], rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Std comparison
        ch1_stds = [s['ch1_std'] for s in file_stats]
        ch2_stds = [s['ch2_std'] for s in file_stats]
        
        axes[0,1].bar(x_pos - 0.2, ch1_stds, 0.4, label='Channel 1', alpha=0.7)
        axes[0,1].bar(x_pos + 0.2, ch2_stds, 0.4, label='Channel 2', alpha=0.7)
        axes[0,1].set_xlabel('Files')
        axes[0,1].set_ylabel('Std Dev (ADC units)')
        axes[0,1].set_title('Standard Deviation Comparison')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f'F{i+1}' for i in range(len(file_names))], rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Range comparison
        ch1_ranges = [s['ch1_range'] for s in file_stats]
        ch2_ranges = [s['ch2_range'] for s in file_stats]
        
        axes[1,0].bar(x_pos - 0.2, ch1_ranges, 0.4, label='Channel 1', alpha=0.7)
        axes[1,0].bar(x_pos + 0.2, ch2_ranges, 0.4, label='Channel 2', alpha=0.7)
        axes[1,0].set_xlabel('Files')
        axes[1,0].set_ylabel('Range (ADC units)')
        axes[1,0].set_title('Signal Range Comparison')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels([f'F{i+1}' for i in range(len(file_names))], rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # File info table
        axes[1,1].axis('off')
        table_text = "FILE DETAILS:\n\n"
        for i, stats in enumerate(file_stats):
            table_text += f"File {i+1}: {stats['name']}\n"
        
        axes[1,1].text(0.1, 0.9, table_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✅ Comparison PDF created: {output_pdf}")
    print(f"   • Analyzed {num_files} files")
    
    # Print summary to console
    print(f"\n📊 Quick Summary:")
    for i, (data, meta) in enumerate(zip(data_list, metadata_list)):
        print(f"   File {i+1}: {Path(meta['path']).name}")
        print(f"     Shape: {data.shape}, Duration: {data.shape[0]/sampling_rate:.1f}s")
        print(f"     Ch1 - Mean: {np.mean(data[:, 0]):.2f}, Std: {np.std(data[:, 0]):.2f}")
        print(f"     Ch2 - Mean: {np.mean(data[:, 1]):.2f}, Std: {np.std(data[:, 1]):.2f}")


def main():
    """Main function to run the inspection."""
    print("🔍 Multiple File Inspector")
    print("=" * 60)
    
    # Parameters
    condition = "healthy"  # Can change to other conditions
    max_files = 3         # Number of files to compare
    window_duration = 0.5  # Window to display (seconds)
    
    print(f"📋 Configuration:")
    print(f"   • Condition: {condition}")
    print(f"   • Max files: {max_files}")
    print(f"   • Window duration: {window_duration}s")
    print()
    
    try:
        inspect_multiple_files(
            output_pdf=f"file_comparison_{condition}.pdf",
            condition=condition,
            max_files=max_files,
            window_duration=window_duration
        )
        
        print(f"🎉 Success! Check the PDF for detailed comparison.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()