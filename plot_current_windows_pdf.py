"""
Script to load a single current signal file, extract 0.5-second windows, 
and plot them to a PDF file.

This script demonstrates:
1. Loading a single current signal file
2. Using the existing windowing module to extract 0.5-second windows
3. Plotting multiple windows in a PDF document
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import DatasetManager, DataLoader, create_windows_for_ml


def plot_current_windows_to_pdf(output_pdf="current_signal_windows.pdf", 
                               condition="healthy", 
                               window_duration=0.5,
                               sampling_rate=10000,
                               max_windows=20):
    """
    Load a single current signal file and plot 0.5-second windows to PDF.
    
    Args:
        output_pdf: Output PDF filename
        condition: Condition to load (healthy, faulty_bearing, etc.)
        window_duration: Duration of each window in seconds (default 0.5s)
        sampling_rate: Sampling rate in Hz (default 10kHz)
        max_windows: Maximum number of windows to plot
    """
    
    print(f"🔍 Loading current signal data for condition: {condition}")
    print("=" * 60)
    
    # Initialize data loader
    dataset_path = Path("data_set")
    data_loader = DataLoader(dataset_path)
    
    # Load a single current signal file
    data_list, metadata_list = data_loader.load_batch(
        condition=condition,
        sensor_type="current",
        max_workers=1
    )
    
    if not data_list:
        print(f"❌ No current signal files found for condition: {condition}")
        return
    
    # Use the first file
    signal_data = data_list[0]  # Shape: [samples, channels]
    file_metadata = metadata_list[0]
    
    print(f"📁 Loaded file: {file_metadata['path']}")
    print(f"📊 Signal shape: {signal_data.shape} (samples, channels)")
    print(f"⏱️  Duration: {signal_data.shape[0] / sampling_rate:.2f} seconds")
    
    # Calculate window parameters
    window_samples = int(window_duration * sampling_rate)
    num_channels = signal_data.shape[1]
    
    print(f"🪟 Window parameters:")
    print(f"   • Window duration: {window_duration}s ({window_samples} samples)")
    print(f"   • Number of channels: {num_channels}")
    
    # Use the existing windowing module to extract windows
    print("🔄 Extracting windows using ml_toolbox...")
    X_windows, labels, win_metadata = create_windows_for_ml(
        [signal_data],  # Pass as list since that's what the function expects
        [file_metadata],  # Pass metadata as list
        window_size=window_samples,
        overlap_ratio=0.5,  # 50% overlap
        balance_classes=False,  # Don't balance since we only have one file
        max_windows_per_class=max_windows
    )
    
    if len(X_windows) == 0:
        print("❌ No windows were extracted")
        return
    
    print(f"✅ Extracted {X_windows.shape[0]} windows")
    print(f"   • Window shape: {X_windows.shape[1:]} (samples, channels)")
    
    # Calculate window start times for labeling
    step_size = window_samples // 2  # 50% overlap
    window_start_times = [i * step_size / sampling_rate for i in range(len(X_windows))]
    
    # Create PDF with plots
    print(f"📄 Creating PDF: {output_pdf}")
    
    with PdfPages(output_pdf) as pdf:
        # Set up the plot style
        plt.style.use('default')
        
        # Plot each window
        for i, (window, start_time) in enumerate(zip(X_windows, window_start_times)):
            
            # Create figure with subplots for each channel
            if num_channels == 1:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                axes = [ax]  # Make it a list for consistency
            else:
                fig, axes = plt.subplots(num_channels, 1, figsize=(12, 8), sharex=True)
                if not isinstance(axes, (list, np.ndarray)):
                    axes = [axes]
            
            # Time axis for the window
            time_axis = np.linspace(0, window_duration, window_samples)
            
            # Plot each channel
            for ch in range(num_channels):
                ax = axes[ch] if num_channels > 1 else axes[0]
                ax.plot(time_axis * 1000, window[:, ch], 'b-', linewidth=0.8)  # Convert to milliseconds
                ax.set_ylabel(f'Current Ch{ch+1}\n(ADC units)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, window_duration * 1000)
                
                # Add some statistics as text
                mean_val = np.mean(window[:, ch])
                std_val = np.std(window[:, ch])
                rms_val = np.sqrt(np.mean(window[:, ch]**2))
                
                # Position text box
                ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nRMS: {rms_val:.1f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set common labels
            axes[-1].set_xlabel('Time (ms)', fontsize=12)
            
            # Title with window information
            end_time = start_time + window_duration
            fig.suptitle(f'Current Signal Window {i+1}/{len(X_windows)}\n'
                        f'File: {Path(file_metadata["path"]).name}\n'
                        f'Time: {start_time:.3f}s - {end_time:.3f}s | '
                        f'Condition: {file_metadata["condition"]} | '
                        f'Load: {file_metadata["load"]} | '
                        f'Frequency: {file_metadata["frequency"]}',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make room for title
            
            # Save to PDF
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"   ✓ Plotted {i + 1}/{len(X_windows)} windows")
        
        # Add a summary page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Calculate total samples from original signal
        total_samples = signal_data.shape[0]
        
        # Summary information
        summary_text = f"""
        CURRENT SIGNAL ANALYSIS SUMMARY
        {'=' * 50}
        
        📁 File Information:
           • Path: {file_metadata['path']}
           • Condition: {file_metadata['condition']}
           • Load: {file_metadata['load']} 
           • Frequency: {file_metadata['frequency']}
        
        📊 Signal Properties:
           • Total samples: {total_samples:,}
           • Total duration: {total_samples / sampling_rate:.2f} seconds
           • Sampling rate: {sampling_rate:,} Hz
           • Number of channels: {num_channels}
        
        🪟 Window Analysis:
           • Window duration: {window_duration}s ({window_samples} samples)
           • Windows extracted: {len(X_windows)}
           • Overlap: 50%
           • Coverage: {len(X_windows) * window_duration / 2:.1f}s of signal
        
        📈 Signal Statistics (All Channels):
           • Mean: {np.mean(signal_data):.2f} ± {np.std(np.mean(signal_data, axis=0)):.2f}
           • Std: {np.mean(np.std(signal_data, axis=0)):.2f} ± {np.std(np.std(signal_data, axis=0)):.2f}
           • RMS: {np.sqrt(np.mean(signal_data**2)):.2f}
           • Min: {np.min(signal_data):.2f}
           • Max: {np.max(signal_data):.2f}
        
        📄 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.title('Analysis Summary', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✅ PDF created successfully: {output_pdf}")
    print(f"   • {len(X_windows)} windows plotted")
    print(f"   • File size: {Path(output_pdf).stat().st_size / 1024:.1f} KB")
    
    return output_pdf


def main():
    """Main function to run the script."""
    print("🚀 Current Signal Window PDF Generator")
    print("=" * 60)
    
    # Available conditions - you can modify this
    available_conditions = ["healthy", "faulty_bearing", "misalignment", "system_misalignment"]
    
    # Parameters - modify as needed
    condition = "system_misalignment"  # Change this to any available condition
    window_duration = 2  # 2 seconds
    max_windows = 15       # Number of windows to plot
    
    print(f"📋 Configuration:")
    print(f"   • Condition: {condition}")
    print(f"   • Window duration: {window_duration}s")
    print(f"   • Max windows: {max_windows}")
    print()
    
    # Generate the PDF
    try:
        output_file = plot_current_windows_to_pdf(
            output_pdf=f"current_windows_{condition}.pdf",
            condition=condition,
            window_duration=window_duration,
            max_windows=max_windows
        )
        
        print(f"🎉 Success! Open '{output_file}' to view the current signal windows.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()