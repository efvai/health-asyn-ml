"""Shared helpers for the Streamlit app."""

import json
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

    # For GDrive datasets, use lazy dataset manager so .dat files are downloaded on demand
    _dm = None
    if (Path(dataset_path) / GDriveCache.MANIFEST_FILENAME).exists():
        _cache = GDriveCache(Path(dataset_path))
        _dm = LazyGDriveDatasetManager(Path(dataset_path), _cache)

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
        dataset_manager=_dm,
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
    # For GDrive datasets, swap in the lazy manager so stubs are downloaded on access
    if (Path(dataset_path) / GDriveCache.MANIFEST_FILENAME).exists():
        _cache = GDriveCache(Path(dataset_path))
        loader.dataset_manager = LazyGDriveDatasetManager(Path(dataset_path), _cache)  # type: ignore[assignment]
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


# ── Google Drive lazy dataset cache ──────────────────────────────────────────

class GDriveCache:
    """Manifest-based on-demand file cache for a public Google Drive folder.

    Workflow
    --------
    1. ``build_manifest(folder_url)`` — scans the remote folder via
       ``gdown.download_folder(skip_download=True)`` and stores a JSON manifest
       mapping ``relative_path → gdrive_file_id``.  No data is downloaded.
    2. ``warm_meta_files()`` — eagerly downloads every ``meta.json`` (tiny,
       needed by the DatasetManager index) and creates 0-byte stub files for
       every ``.dat`` signal file so that ``scan_dataset()`` can enumerate them.
    3. ``ensure_file(rel_path)`` — downloads a single file on demand the first
       time it is needed (called by :class:`LazyGDriveDatasetManager`).
    """

    MANIFEST_FILENAME = "_gdrive_manifest.json"
    # Class-level throttle: minimum seconds between gdown calls to avoid rate limiting
    _MIN_DOWNLOAD_INTERVAL: float = 1.5
    _last_download_time: float = 0.0

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, str] = {}
        self._load_manifest()

    # ── Manifest persistence ──────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.cache_dir / self.MANIFEST_FILENAME

    def _load_manifest(self) -> bool:
        mp = self._manifest_path()
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                self._manifest = json.load(f)
            return True
        return False

    def _save_manifest(self) -> None:
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def has_manifest(self) -> bool:
        return bool(self._manifest)

    # ── Core API ──────────────────────────────────────────────────────────────

    def build_manifest(self, folder_url: str) -> None:
        """Scan the remote folder and save the file manifest.  No files downloaded."""
        import gdown
        from gdown.download_folder import download_folder as _gdown_folder

        # skip_download=True → returns GoogleDriveFileToDownload(id, path, local_path)
        # Pass output=cache_dir (no trailing sep) so local_path = cache_dir/path
        entries = _gdown_folder(
            url=folder_url,
            output=str(self.cache_dir),
            skip_download=True,
            quiet=True,
        )
        if not entries:
            raise RuntimeError(
                "Google Drive returned an empty file listing. "
                "Make sure the folder is shared publicly (Anyone with link)."
            )

        self._manifest = {}
        for entry in entries:
            # When skip_download=True, entries are GoogleDriveFileToDownload namedtuples
            entry_local = getattr(entry, "local_path", None)
            entry_id = getattr(entry, "id", None)
            entry_rel = getattr(entry, "path", None)
            if entry_local is None or entry_id is None:
                continue  # folder stub or unexpected type
            local = Path(entry_local)
            try:
                rel = str(local.relative_to(self.cache_dir))
            except ValueError:
                rel = str(entry_rel) if entry_rel is not None else str(local.name)
            self._manifest[rel] = entry_id

        self._save_manifest()

    def warm_meta_files(self, progress_fn=None) -> int:
        """Download all meta.json files and create 0-byte stubs for .dat files.

        Returns the number of meta files downloaded.
        """
        meta_keys = [k for k in self._manifest if k.endswith("meta.json")]
        dat_keys = [k for k in self._manifest if k.lower().endswith(".dat")]

        # Download meta files
        for i, rel_path in enumerate(meta_keys):
            self.ensure_file(rel_path)
            if progress_fn:
                progress_fn(i + 1, len(meta_keys))

        # Create 0-byte stubs so scan_dataset() can enumerate signal files
        for rel_path in dat_keys:
            local = self.cache_dir / rel_path
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                local.touch()

        return len(meta_keys)

    def ensure_file(self, rel_path: str, max_retries: int = 3) -> Path:
        """Download file if missing or empty stub.  Returns local path.

        Retries up to *max_retries* times with exponential back-off to handle
        transient Google Drive rate-limiting.  On total failure raises
        ``RuntimeError`` with a human-readable message and the direct browser
        URL so the file can be fetched manually.
        """
        import time
        from gdown.download import download as _gdown_dl
        from gdown.exceptions import FileURLRetrievalError as _GDownURLError

        local = self.cache_dir / rel_path
        if local.exists() and local.stat().st_size > 0:
            return local

        # Normalise separators when looking up the manifest
        file_id: Optional[str] = self._manifest.get(rel_path)
        if file_id is None:
            rel_norm = rel_path.replace("\\", "/")
            for k, v in self._manifest.items():
                if k.replace("\\", "/") == rel_norm:
                    file_id = v
                    break
        if file_id is None:
            raise FileNotFoundError(
                f"'{rel_path}' is not in the GDrive manifest. "
                "Run build_manifest() first."
            )

        local.parent.mkdir(parents=True, exist_ok=True)
        browser_url = f"https://drive.google.com/uc?id={file_id}"
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(max_retries):
            # Enforce minimum inter-download interval to avoid Google rate limiting
            now = time.time()
            wait = GDriveCache._MIN_DOWNLOAD_INTERVAL - (now - GDriveCache._last_download_time)
            if wait > 0:
                time.sleep(wait)
            if attempt > 0:
                time.sleep(2 ** attempt)  # extra back-off: 2 s, 4 s, …
            # Alternate cookie usage: cookies off on odd attempts (different session path)
            use_cookies = (attempt % 2 == 0)
            try:
                GDriveCache._last_download_time = time.time()
                _gdown_dl(id=file_id, output=str(local), quiet=True, use_cookies=use_cookies)
                # gdown may leave a 0-byte file on failure; verify
                if local.stat().st_size == 0:
                    raise _GDownURLError("downloaded file is empty")
                return local
            except _GDownURLError as exc:
                last_exc = exc
                # Remove empty stub so it can be retried cleanly
                if local.exists() and local.stat().st_size == 0:
                    local.unlink()

        raise RuntimeError(
            f"Google Drive download failed for '{rel_path}' after "
            f"{max_retries} attempts.\n\n"
            "Possible causes:\n"
            "  • The file sharing is not set to 'Anyone with the link' — "
            "check permissions in Google Drive.\n"
            "  • Google is rate-limiting gdown (too many recent requests) — "
            "wait a few minutes and retry.\n\n"
            f"You can try downloading the file manually from:\n"
            f"  {browser_url}\n\n"
            f"Last gdown error: {last_exc}"
        ) from last_exc

    def ensure_absolute(self, abs_path: Path) -> Path:
        """Like ensure_file but accepts an absolute path inside cache_dir."""
        try:
            rel = str(abs_path.relative_to(self.cache_dir))
        except ValueError:
            raise ValueError(f"{abs_path} is not inside cache_dir {self.cache_dir}")
        return self.ensure_file(rel)

    def cached_count(self) -> int:
        """Number of real (non-stub) files currently on disk."""
        return sum(
            1 for rel in self._manifest
            if (self.cache_dir / rel).exists()
            and (self.cache_dir / rel).stat().st_size > 0
        )

    def total_count(self) -> int:
        return len(self._manifest)


class LazyGDriveDatasetManager:
    """Drop-in replacement for DatasetManager that downloads .dat files on demand.

    Delegates everything to the wrapped DatasetManager; only ``load_sample``
    is intercepted to trigger a download when the file is missing or is a
    0-byte stub.
    """

    def __init__(self, dataset_path: Path, cache: GDriveCache):
        ensure_toolbox_on_path()
        from ml_toolbox.data_loader.dataset_manager import DatasetManager
        self._dm = DatasetManager(dataset_path)
        self._cache = cache

    # Forward attribute access to the wrapped DatasetManager
    def __getattr__(self, name: str):
        return getattr(self._dm, name)

    def load_sample(self, file_info: Dict) -> Any:
        """Ensure the file is on disk, then delegate to the real loader."""
        abs_path = Path(file_info["absolute_path"])
        if not abs_path.exists() or abs_path.stat().st_size == 0:
            self._cache.ensure_absolute(abs_path)
        return self._dm.load_sample(file_info)


def connect_gdrive_dataset(
    folder_url: str,
    local_name: str,
    project_root: Path,
    progress_fn=None,
) -> Path:
    """Build a GDrive manifest and warm meta files for *local_name* dataset.

    Parameters
    ----------
    folder_url:
        Public Google Drive folder URL.
    local_name:
        Name of the local dataset folder (e.g. ``"data_set_gdrive"``).
    project_root:
        Project root directory.
    progress_fn:
        Optional ``fn(step: str, done: int, total: int)`` for progress updates.

    Returns
    -------
    Path
        Local path to the dataset folder.
    """
    cache_dir = project_root / local_name
    cache = GDriveCache(cache_dir)
    cache.build_manifest(folder_url)
    cache.warm_meta_files(
        progress_fn=lambda done, tot: (
            progress_fn("meta", done, tot) if progress_fn else None
        )
    )
    return cache_dir
