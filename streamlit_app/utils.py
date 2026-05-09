"""Shared helpers for the Streamlit app."""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
    n_points: int = 4000,
) -> plt.Figure:
    """Side-by-side time-domain and frequency-domain before/after plot."""
    from scipy.signal import welch

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if filtered.ndim == 1:
        filtered = filtered[:, np.newaxis]

    n = min(n_points, len(raw))
    t = np.arange(n) / fs

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Time domain
    axes[0].plot(t, raw[:n, channel], color="#2196F3", linewidth=0.7, alpha=0.85, label="Raw")
    axes[0].plot(t, filtered[:n, channel], color="#F44336", linewidth=0.7, alpha=0.85, label="Filtered")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Time domain — ch{channel + 1}")
    axes[0].legend()

    # Frequency domain (Welch PSD)
    nperseg = min(4096, len(raw))
    f_r, p_r = welch(raw[:, channel].astype(np.float64), fs=fs, nperseg=nperseg)
    f_f, p_f = welch(filtered[:, channel].astype(np.float64), fs=fs, nperseg=nperseg)
    axes[1].semilogy(f_r, p_r, color="#2196F3", linewidth=0.8, alpha=0.85, label="Raw")
    axes[1].semilogy(f_f, p_f, color="#F44336", linewidth=0.8, alpha=0.85, label="Filtered")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title(f"Frequency domain (Welch PSD) — ch{channel + 1}")
    axes[1].legend()

    plt.tight_layout()
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


# ── Google Drive downloader ───────────────────────────────────────────────────

def download_from_gdrive(url: str, dest_name: str, project_root: Path) -> Path:
    """Download a dataset from a Google Drive shared link.

    Supports:
    - Shared **folder** links  → downloaded recursively into *dest_name/*
    - Shared **file** links    → downloaded; ZIP archives are extracted into *dest_name/*

    Returns the local :class:`Path` to the dataset folder.

    Raises
    ------
    ImportError
        If ``gdown`` is not installed.
    ValueError
        If the URL is not recognised or the downloaded file is not a ZIP.
    RuntimeError
        If gdown downloaded nothing.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is not installed.\n"
            "Install it with:  pip install gdown"
        )

    import re
    import shutil
    import tempfile
    import zipfile

    url = url.strip()
    dest_path = project_root / dest_name

    # ── Classify URL ──────────────────────────────────────────────────────────
    folder_match = re.search(r"/drive/folders/([A-Za-z0-9_-]+)", url)
    file_match   = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    id_match     = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)

    if folder_match:
        # ── Google Drive folder ───────────────────────────────────────────────
        folder_id = folder_match.group(1)
        with tempfile.TemporaryDirectory() as tmp_parent:
            gdown.download_folder(
                id=folder_id,
                output=tmp_parent,
                quiet=False,
            )
            # gdown places files under tmp_parent/FOLDER_NAME/
            children = list(Path(tmp_parent).iterdir())
            if not children:
                raise RuntimeError("gdown did not download any files from the folder.")
            src = children[0] if (len(children) == 1 and children[0].is_dir()) else Path(tmp_parent)
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(str(src), str(dest_path))
    else:
        # ── Single file (ZIP expected) ────────────────────────────────────────
        file_id = None
        if file_match:
            file_id = file_match.group(1)
        elif id_match:
            file_id = id_match.group(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".download") as tmp_f:
            tmp_path = Path(tmp_f.name)
        try:
            gdown.download(
                id=file_id if file_id else url,
                output=str(tmp_path),
                quiet=False,
                fuzzy=True,
            )
            if not zipfile.is_zipfile(tmp_path):
                raise ValueError(
                    "The downloaded file is not a ZIP archive. "
                    "Please share a Google Drive *folder* link, or a ZIP file link."
                )
            if dest_path.exists():
                shutil.rmtree(dest_path)
            dest_path.mkdir(parents=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(dest_path)
            # Flatten single-root-folder zip so dest_path/ contains the data directly
            children = list(dest_path.iterdir())
            if len(children) == 1 and children[0].is_dir():
                inner = children[0]
                for item in inner.iterdir():
                    shutil.move(str(item), str(dest_path / item.name))
                inner.rmdir()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return dest_path
