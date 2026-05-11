"""Shared helpers for the Streamlit app."""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── HuggingFace Hub ───────────────────────────────────────────────────────────

HF_REPO_ID = "efvai/health-asyn-current-vibro-data"


@st.cache_data(show_spinner="Fetching dataset list from Hugging Face…", ttl=3600)
def list_hf_datasets(repo_id: str = HF_REPO_ID) -> List[str]:
    """Return sorted list of data_set_* top-level dirs in the HF repo."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    dirs = {f.split("/")[0] for f in files if "/" in f and f.split("/")[0].startswith("data_set_")}
    return sorted(dirs)


def download_hf_dataset(dataset_name: str, project_root: Path, repo_id: str = HF_REPO_ID) -> Path:
    """Download *dataset_name* from HF into project_root, mirroring the repo structure."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{dataset_name}/**"],
        local_dir=str(project_root),
    )
    return project_root / dataset_name

# ── Path helpers ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the project root (parent of streamlit_app/)."""
    return Path(__file__).parent.parent


def ensure_toolbox_on_path():
    """Add project root to sys.path so ml_toolbox is importable."""
    root = str(get_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def discover_datasets(root: Optional[Path] = None) -> List[str]:
    """Return sorted list of data_set_* folder names found at *root*."""
    if root is None:
        root = get_project_root()
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("data_set"))


# ── DataLoader helpers ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Reading dataset index…")
def get_loader_index(dataset_path: str) -> Dict[str, Any]:
    """Cache the dataset index (cheap metadata scan, not signal loading)."""
    ensure_toolbox_on_path()
    from ml_toolbox.data_loader import DataLoader
    loader = DataLoader(Path(dataset_path))
    return loader.index


def get_filtered_file_list(
    dataset_path: str,
    classes=None,        # tuple[str] or None
    loads=None,          # tuple[float] or None
    frequencies=None,    # tuple[float] or None
    sensor_types=None,   # tuple[str] or None
) -> List[Dict]:
    """Return file-info dicts matching the given filter criteria.

    Uses the already-cached index (no signal loading).  Filtering is done in
    pure Python so this is instant on every render.
    """
    idx = get_loader_index(dataset_path)
    files = idx.get("files", [])
    if classes:
        classes_set = set(classes)
        files = [f for f in files if f.get("class") in classes_set]
    if loads:
        loads_set = {float(v) for v in loads}
        files = [f for f in files if float(f.get("load", "nan")) in loads_set]
    if frequencies:
        freqs_set = {float(v) for v in frequencies}
        files = [f for f in files if float(f.get("electrical_frequency_hz", "nan")) in freqs_set]
    if sensor_types:
        st_set = set(sensor_types)
        files = [f for f in files if f.get("sensor_type") in st_set]
    return files


def run_lazy_extraction(
    dataset_path: str,
    file_list: List[Dict],
    lpf_enabled: bool,
    lpf_cutoff: float,
    lpf_order: int,
    window_size: int,
    overlap_ratio: float,
    shuffle: bool,
    random_state: Optional[int],
    sensor_type: str,
    feature_config,
    class_to_int: Optional[Dict] = None,
    progress_bar=None,
    status_text=None,
):
    """Run extract_features_lazy with an optional Streamlit progress bar.

    Returns (features, labels, feature_names, win_metadata, label_map).
    """
    ensure_toolbox_on_path()
    from ml_toolbox.data_loader import extract_features_lazy
    from ml_toolbox import ButterworthLPF

    preprocessor = ButterworthLPF(cutoff_hz=lpf_cutoff, order=lpf_order) if lpf_enabled else None

    def _progress(done: int, tot: int):
        if progress_bar is not None:
            progress_bar.progress(done / tot)
        if status_text is not None:
            status_text.text(f"Processing file {done} / {tot}…")

    return extract_features_lazy(
        dataset_path=Path(dataset_path),
        file_list=file_list,
        preprocessor=preprocessor,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        shuffle=shuffle,
        random_state=random_state,
        sensor_type=sensor_type,
        feature_config=feature_config,
        class_to_int=class_to_int,
        progress_callback=_progress,
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────

def class_distribution_chart(metadata_list: List[Dict]) -> plt.Figure:
    """Bar chart of sample count per class."""
    from collections import Counter
    counts = Counter(m["class"] for m in metadata_list)
    df = pd.DataFrame(counts.items(), columns=["class", "count"]).sort_values("class")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["class"], df["count"], color="#2196F3")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.set_title("Samples per class")
    plt.tight_layout()
    return fig


def signal_preview_chart(data: np.ndarray, metadata: Dict, n_points: int = 4000) -> plt.Figure:
    """Line plot of the first *n_points* samples for each channel in *data*."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_ch = data.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    x = np.arange(min(n_points, len(data)))
    for i, ax in enumerate(axes):
        ax.plot(x, data[: len(x), i], linewidth=0.6, color="#2196F3")
        ax.set_ylabel(f"ch{i + 1}")
    axes[-1].set_xlabel("Sample index")
    label = metadata.get("class", "?")
    freq = metadata.get("electrical_frequency_hz", "?")
    load = metadata.get("load", "?")
    fig.suptitle(f"Class={label}  freq={freq} Hz  load={load}", fontsize=9)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Loading sample for preview…")
def cached_load_single_raw(dataset_path: str, sample_id: str, sensor_type: str):
    """Load a single sample without any preprocessing (for before/after preview)."""
    ensure_toolbox_on_path()
    from ml_toolbox.data_loader import DataLoader
    loader = DataLoader(Path(dataset_path))
    data_list, metadata_list = loader.load_batch(
        sample_ids=[sample_id],
        sensor_types=[sensor_type] if sensor_type else None,
        preprocessor=None,
    )
    if not data_list:
        return None, None
    arr = data_list[0]
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return arr.astype(np.float32), metadata_list[0]


def preprocessing_preview_chart(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    channel: int,
    freq_mode: str = "Welch",
    nperseg: int = 4096,
):
    """Interactive side-by-side time-domain and frequency-domain before/after plot (plotly)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if filtered.ndim == 1:
        filtered = filtered[:, np.newaxis]

    t = np.arange(len(raw)) / fs

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Time domain — ch{channel + 1}",
            f"{freq_mode} spectrum — ch{channel + 1}",
        ),
    )

    # Time domain — WebGL for large arrays
    fig.add_trace(
        go.Scattergl(x=t, y=raw[:, channel], name="Raw",
                     line=dict(color="#2196F3", width=1), opacity=0.85),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=t, y=filtered[:, channel], name="Filtered",
                     line=dict(color="#F44336", width=1), opacity=0.85),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    # Frequency domain
    sig_raw = raw[:, channel].astype(np.float64)
    sig_fil = filtered[:, channel].astype(np.float64)
    if freq_mode == "Welch":
        from scipy.signal import welch as _welch
        _nperseg = min(nperseg, len(sig_raw))
        f_r, p_r = _welch(sig_raw, fs=fs, nperseg=_nperseg)
        f_f, p_f = _welch(sig_fil, fs=fs, nperseg=_nperseg)
        y_label = "PSD"
    else:  # FFT with Hann window
        _win_r = np.hanning(len(sig_raw))
        _win_f = np.hanning(len(sig_fil))
        f_r = np.fft.rfftfreq(len(sig_raw), d=1.0 / fs)
        p_r = np.abs(np.fft.rfft(sig_raw * _win_r))
        f_f = np.fft.rfftfreq(len(sig_fil), d=1.0 / fs)
        p_f = np.abs(np.fft.rfft(sig_fil * _win_f))
        y_label = "Magnitude"

    fig.add_trace(
        go.Scatter(x=f_r, y=p_r, name="Raw", showlegend=False,
                   line=dict(color="#2196F3", width=1), opacity=0.85),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=f_f, y=p_f, name="Filtered", showlegend=False,
                   line=dict(color="#F44336", width=1), opacity=0.85),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text=y_label, type="log", row=1, col=2)

    fig.update_layout(
        height=450,
        margin=dict(t=50, b=10, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.25),
    )
    return fig


def time_domain_chart(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    channel: int,
    win_start: int,
    win_size: int,
):
    """Full-signal time-domain plot with a shaded rectangle over the current window (Plotly).

    raw / filtered : (n_samples, n_channels) float arrays.
    The yellow band marks [win_start, win_start + win_size).
    """
    import plotly.graph_objects as go

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if filtered.ndim == 1:
        filtered = filtered[:, np.newaxis]

    n = len(raw)
    t = np.arange(n) / fs
    t_start = win_start / fs
    t_end = min(win_start + win_size, n) / fs

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=raw[:, channel], name="Raw",
        line=dict(color="#2196F3", width=1), opacity=0.8,
    ))
    fig.add_trace(go.Scattergl(
        x=t, y=filtered[:, channel], name="Filtered",
        line=dict(color="#F44336", width=1), opacity=0.8,
    ))
    fig.add_vrect(
        x0=t_start, x1=t_end,
        fillcolor="rgba(255, 193, 7, 0.25)",
        layer="below", line_width=0,
        annotation_text="window", annotation_position="top left",
        annotation_font_size=11,
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(
        title=f"Time domain — ch{channel + 1}",
        height=320,
        margin=dict(t=45, b=10, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def window_frequency_chart(
    raw_window: np.ndarray,
    filtered_window: np.ndarray,
    fs: float,
    channel: int,
    freq_mode: str = "Welch",
    nperseg: int = 4096,
):
    """Frequency-domain plot for a single pre-sliced window (Plotly).

    raw_window / filtered_window : (win_size, n_channels) float arrays.
    Applies Welch PSD or FFT-with-Hann exactly as the feature extraction pipeline does.
    """
    import plotly.graph_objects as go

    if raw_window.ndim == 1:
        raw_window = raw_window[:, np.newaxis]
    if filtered_window.ndim == 1:
        filtered_window = filtered_window[:, np.newaxis]

    sig_r = raw_window[:, channel].astype(np.float64)
    sig_f = filtered_window[:, channel].astype(np.float64)

    if freq_mode == "Welch":
        from scipy.signal import welch as _welch
        _nperseg = min(nperseg, len(sig_r))
        f_r, p_r = _welch(sig_r, fs=fs, nperseg=_nperseg)
        f_f, p_f = _welch(sig_f, fs=fs, nperseg=_nperseg)
        y_label = "PSD"
    else:  # FFT with Hann window
        hann_r = np.hanning(len(sig_r))
        hann_f = np.hanning(len(sig_f))
        f_r = np.fft.rfftfreq(len(sig_r), d=1.0 / fs)
        p_r = np.abs(np.fft.rfft(sig_r * hann_r))
        f_f = np.fft.rfftfreq(len(sig_f), d=1.0 / fs)
        p_f = np.abs(np.fft.rfft(sig_f * hann_f))
        y_label = "Magnitude"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_r, y=p_r, name="Raw",
        line=dict(color="#2196F3", width=1), opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=f_f, y=p_f, name="Filtered",
        line=dict(color="#F44336", width=1), opacity=0.85,
    ))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text=y_label, type="log")
    fig.update_layout(
        title=f"{freq_mode} spectrum — ch{channel + 1}  (window only)",
        height=350,
        margin=dict(t=45, b=10, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def window_signal_chart(window: np.ndarray, channel: int, fs: float = 1.0, title: str = "") -> plt.Figure:
    """Time-domain and Welch PSD side-by-side for a single window (window_size, n_channels)."""
    from scipy.signal import welch

    signal = window[:, channel].astype(np.float64)
    t = np.arange(len(signal)) / fs
    nperseg = min(4096, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(t, signal, color="#4CAF50", linewidth=0.7)
    axes[0].set_xlabel("Time (s)" if fs > 1.0 else "Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Time domain — ch{channel + 1}" + (f"  |  {title}" if title else ""))

    axes[1].semilogy(freqs, psd, color="#4CAF50", linewidth=0.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title(f"Welch PSD — ch{channel + 1}" + (f"  |  {title}" if title else ""))

    plt.tight_layout()
    return fig


def confusion_matrix_chart(cm: np.ndarray, class_names: List[str]) -> plt.Figure:
    """Heatmap of a confusion matrix."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.2), max(4, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def correlation_heatmap(features: np.ndarray, feature_names: List[str]) -> plt.Figure:
    """Seaborn correlation heatmap for a feature matrix."""
    import seaborn as sns
    df = pd.DataFrame(features, columns=feature_names)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.6), max(6, len(feature_names) * 0.5)))
    sns.heatmap(corr, annot=len(feature_names) <= 20, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig
