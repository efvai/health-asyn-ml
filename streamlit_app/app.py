"""
Streamlit ML Toolbox Pipeline Debugger
Run: streamlit run streamlit_app/app.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is on path before any ml_toolbox import
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import state as S
from utils import (
    get_project_root,
    discover_datasets,
    get_loader_index,
    get_filtered_file_list,
    run_lazy_extraction,
    cached_load_single_raw,
    class_distribution_chart,
    window_signal_chart,
    confusion_matrix_chart,
    correlation_heatmap,
    time_domain_chart,
    window_frequency_chart,
    HF_REPO_ID,
    list_hf_datasets,
    download_hf_dataset,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Debugger",
    page_icon="🔬",
    layout="wide",
)
st.title("ML Pipeline Debugger")

PROJECT_ROOT = get_project_root()
AVAILABLE_DATASETS = discover_datasets(PROJECT_ROOT)

# ── Shared defaults — set once before any tab renders ───────────────────────
for _key, _val in [
    ("lpf_enabled", True), ("lpf_cutoff", 500), ("lpf_order", 4),
    ("win_size", 10000), ("win_overlap", 0.5),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — dataset selector (shared across tabs)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Datasets")

    train_dataset = st.selectbox(
        "Train dataset",
        AVAILABLE_DATASETS,
        index=AVAILABLE_DATASETS.index("data_set_4") if "data_set_4" in AVAILABLE_DATASETS else 0,
        key="sidebar_train_dataset",
    )
    test_dataset = st.selectbox(
        "Test dataset",
        AVAILABLE_DATASETS,
        index=AVAILABLE_DATASETS.index("data_set_6") if "data_set_6" in AVAILABLE_DATASETS else 0,
        key="sidebar_test_dataset",
    )

    train_path = str(PROJECT_ROOT / train_dataset) if train_dataset else None
    test_path = str(PROJECT_ROOT / test_dataset) if test_dataset else None

    st.divider()

    with st.expander("Download from Hugging Face", expanded=False):
        st.caption(f"`{HF_REPO_ID}`")
        try:
            _hf_datasets = list_hf_datasets(HF_REPO_ID)
        except Exception as _e:
            _hf_datasets = []
            st.warning(f"Could not reach Hugging Face: {_e}")

        if _hf_datasets:
            _hf_selected = st.selectbox(
                "Dataset",
                _hf_datasets,
                key="hf_dataset_select",
            )
            _local_exists = (PROJECT_ROOT / _hf_selected).exists() if _hf_selected else False
            if _local_exists:
                st.caption(f"✔️ Already downloaded locally.")

            _dl_msg = st.session_state.pop("_hf_dl_msg", None)
            if _dl_msg:
                if _dl_msg["type"] == "success":
                    st.success(_dl_msg["text"])
                else:
                    st.error(_dl_msg["text"])

            if st.button("Download", key="btn_hf_download", disabled=not _hf_selected):
                try:
                    with st.spinner(f"Downloading {_hf_selected} from Hugging Face…"):
                        download_hf_dataset(_hf_selected, PROJECT_ROOT)
                    st.session_state["_hf_dl_msg"] = {
                        "type": "success",
                        "text": f"'{_hf_selected}' downloaded. Select it in the dropdowns above.",
                    }
                except Exception as _e:
                    st.session_state["_hf_dl_msg"] = {
                        "type": "error",
                        "text": f"Download failed: {_e}",
                    }
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Guard — no datasets available
# ─────────────────────────────────────────────────────────────────────────────
if train_path is None or test_path is None:
    st.warning(
        "No datasets found. "
        "Use **Download from Hugging Face** in the sidebar to add one."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_info, tab_prep, tab_features, tab_train, tab_shap = st.tabs([
    "1 · Dataset Info",
    "2 · Preprocessing",
    "3 · Feature Extraction",
    "4 · Train & Evaluate",
    "5 · SHAP",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Info
# ═════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("Dataset Info")

    col_train_info, col_test_info = st.columns(2)

    for _col, _path, _label in [
        (col_train_info, train_path, "Train"),
        (col_test_info, test_path, "Test"),
    ]:
        with _col:
            st.markdown(f"**{_label}: `{Path(_path).name}`**")
            try:
                _idx = get_loader_index(_path)
                _files = _idx.get("files", [])

                _i1, _i2, _i3, _i4 = st.columns(4)
                _i1.metric("Files", len(_files))
                _i2.metric("Classes", len(_idx.get("classes", [])))
                _i3.metric("Sensor types", len(_idx.get("sensor_types", [])))
                _i4.metric("Frequencies", len(_idx.get("electrical_frequencies_hz", [])))

                st.write("**Classes:**", sorted(_idx.get("classes", [])))
                st.write("**Loads:**", sorted(set(_idx.get("loads", []))))
                st.write("**Frequencies (Hz):**", sorted(_idx.get("electrical_frequencies_hz", [])))
                st.write("**Sensor types:**", sorted(_idx.get("sensor_types", [])))

                if _files:
                    _counts = pd.Series([f["class"] for f in _files]).value_counts().sort_index()
                    with st.expander("Files per class", expanded=False):
                        st.dataframe(
                            _counts.rename("count").reset_index().rename(columns={"index": "class"}),
                            width='stretch',
                        )

            except Exception as _e:
                st.error(f"Could not read {_label} index: {_e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
with tab_prep:
    st.subheader("Preprocessing")

    st.markdown("### Butterworth Low-Pass Filter")
    col_lpf1, col_lpf2, col_lpf3 = st.columns(3)
    with col_lpf1:
        st.toggle("Enable LPF", key="lpf_enabled")
    with col_lpf2:
        st.slider("Cutoff (Hz)", min_value=10, max_value=5000, step=10,
                  key="lpf_cutoff", disabled=not st.session_state["lpf_enabled"])
    with col_lpf3:
        st.selectbox("Order", [2, 4, 6, 8],
                     key="lpf_order", disabled=not st.session_state["lpf_enabled"])

    st.info("LPF settings are applied during feature extraction in **3 · Feature Extraction**.")

    # ── Windowing ─────────────────────────────────────────────────────────────
    st.markdown("### Windowing")
    st.caption("Shared with **3 · Feature Extraction**.")
    col_win1, col_win2 = st.columns(2)
    with col_win1:
        st.number_input(
            "Window size (samples)", min_value=256, max_value=100000,
            step=256, key="win_size",
        )
    with col_win2:
        st.slider("Overlap ratio", 0.0, 0.9, step=0.05, key="win_overlap")

    # ── Before / After Preview ────────────────────────────────────────────────
    st.markdown("### Before / After Preview")

    try:
        prep_index = get_loader_index(train_path)
    except Exception as e:
        st.error(f"Could not read index: {e}")
        prep_index = {}

    prep_sensor_types = sorted(prep_index.get("sensor_types", []))
    files_meta = prep_index.get("files", [])

    if prep_sensor_types:
        prep_sensor = st.selectbox(
            "Sensor type", prep_sensor_types,
            index=prep_sensor_types.index("vibration") if "vibration" in prep_sensor_types else 0,
            key="prep_sensor",
        )

        sample_ids_for_sensor = sorted(set(
            f["sample_id"] for f in files_meta
            if f.get("sensor_type") == prep_sensor
        ))
        if not sample_ids_for_sensor:
            sample_ids_for_sensor = sorted(set(f.get("sample_id", "") for f in files_meta))

        # Build per-sample metadata lookup for enriched selectbox labels
        _fs_key = "sample_rate_vibro_hz" if prep_sensor == "vibration" else "sample_rate_current_hz"
        _sample_meta_lut: dict = {}
        for _f in files_meta:
            _sid = _f.get("sample_id", "")
            if _sid and _sid not in _sample_meta_lut and _f.get("sensor_type") == prep_sensor:
                _sample_meta_lut[_sid] = _f

        def _fmt_sample_id(sid: str) -> str:
            _m = _sample_meta_lut.get(sid, {})
            _cls = _m.get("class", "?")
            _ld = _m.get("load", "?")
            _fq = _m.get("electrical_frequency_hz", "?")
            _fv = _m.get(_fs_key, "?")
            return f"{sid} | {_cls} | load={_ld} | {_fq}Hz | fs={_fv}Hz"

        if sample_ids_for_sensor:
            sel_sample_id = st.selectbox(
                "Sample ID", sample_ids_for_sensor, key="prep_sample_id",
                format_func=_fmt_sample_id,
            )

            if st.button("Load Sample for Preview", type="primary", key="btn_prep_preview"):
                raw, raw_meta = cached_load_single_raw(train_path, sel_sample_id, prep_sensor)
                if raw is None:
                    st.error("Could not load sample.")
                else:
                    st.session_state["_prep_raw"] = raw
                    st.session_state["_prep_meta"] = raw_meta
                    st.session_state["prep_win_start"] = 0

            if "_prep_raw" in st.session_state:
                raw = st.session_state["_prep_raw"]
                raw_meta = st.session_state["_prep_meta"]
                if raw.ndim == 1:
                    raw = raw[:, np.newaxis]
                n_ch = raw.shape[1]

                _fs_key_raw = "sample_rate_vibro_hz" if prep_sensor == "vibration" else "sample_rate_current_hz"
                fs = float(raw_meta.get(_fs_key_raw) or 1.0)

                if st.session_state["lpf_enabled"]:
                    from ml_toolbox import ButterworthLPF
                    try:
                        _filt = ButterworthLPF(
                            cutoff_hz=float(st.session_state["lpf_cutoff"]),
                            order=int(st.session_state["lpf_order"]),
                        )
                        filtered = _filt.apply(raw.astype(np.float64), fs=fs).astype(np.float32)
                    except Exception as _e:
                        st.error(f"LPF error: {_e}")
                        filtered = raw
                else:
                    filtered = raw

                # Window parameters (shared with Feature Extraction tab)
                n_samples = len(raw)
                _win_size = int(st.session_state.get("win_size", 10000))
                _overlap = float(st.session_state.get("win_overlap", 0.5))
                _hop_size = max(1, int(_win_size * (1 - _overlap)))
                _max_start = max(0, n_samples - _win_size)

                # Clamp stored window start
                _prep_win_start = max(0, min(
                    int(st.session_state.get("prep_win_start", 0)), _max_start
                ))
                st.session_state["prep_win_start"] = _prep_win_start

                # ── Channel selector ──────────────────────────────────────────
                ch_sel = st.selectbox(
                    "Channel", list(range(n_ch)), format_func=lambda i: f"ch{i + 1}",
                    key="prep_ch_sel",
                )

                # ── Time domain ───────────────────────────────────────────────
                st.markdown("#### Time domain")
                fig_time = time_domain_chart(
                    raw, filtered, fs, ch_sel, _prep_win_start, _win_size,
                )
                st.plotly_chart(fig_time, width='stretch')

                # ── Frequency domain (windowed) ───────────────────────────────
                st.markdown("#### Frequency domain — current window")

                col_fm, col_np = st.columns(2)
                with col_fm:
                    freq_mode = st.radio(
                        "Method", ["Welch", "FFT"],
                        horizontal=True, key="prep_freq_mode",
                    )
                with col_np:
                    nperseg = st.number_input(
                        "nperseg (Welch)", min_value=256, max_value=65536,
                        value=4096, step=256, key="prep_nperseg",
                        disabled=(freq_mode != "Welch"),
                    )

                # Window info metrics
                _win_end = min(_prep_win_start + _win_size, n_samples)
                _total_wins = (_max_start // _hop_size + 1) if _hop_size > 0 else 1
                _cur_idx = _prep_win_start // _hop_size if _hop_size > 0 else 0
                wm1, wm2, wm3 = st.columns(3)
                wm1.metric("Start sample", _prep_win_start)
                wm2.metric("End sample", _win_end)
                wm3.metric("Window", f"{_cur_idx + 1} / {_total_wins}")

                # Navigation buttons + jump input
                nav1, nav2, nav3 = st.columns([1, 1, 2])
                with nav1:
                    if st.button("⬅ Prev", key="btn_win_prev",
                                 disabled=(_prep_win_start <= 0)):
                        st.session_state["prep_win_start"] = max(0, _prep_win_start - _hop_size)
                        st.rerun()
                with nav2:
                    if st.button("Next ➡", key="btn_win_next",
                                 disabled=(_prep_win_start >= _max_start)):
                        st.session_state["prep_win_start"] = min(_max_start, _prep_win_start + _hop_size)
                        st.rerun()
                with nav3:
                    st.number_input(
                        "Jump to sample", min_value=0, max_value=max(0, _max_start),
                        step=_hop_size, key="prep_win_start",
                    )

                # Slice and plot frequency domain for current window
                _s = _prep_win_start
                _e = min(_s + _win_size, n_samples)
                raw_win = raw[_s:_e, :]
                filt_win = filtered[_s:_e, :]

                if len(raw_win) > 0:
                    fig_freq = window_frequency_chart(
                        raw_win, filt_win, fs, ch_sel,
                        freq_mode=freq_mode, nperseg=int(nperseg),
                    )
                    st.plotly_chart(fig_freq, width='stretch')
        else:
            st.info("No samples found in index.")
    else:
        st.info("No sensor types found — check the train dataset in the sidebar.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════
with tab_features:
    st.subheader("Feature Extraction")

    _lpf_status = (
        f"LPF ON — cutoff {st.session_state['lpf_cutoff']} Hz, "
        f"order {st.session_state['lpf_order']}"
        if st.session_state["lpf_enabled"] else "LPF OFF"
    )
    st.caption(f"{_lpf_status} · Configure in **2 · Preprocessing**")

    # ── Data filters ─────────────────────────────────────────────────────────
    st.markdown("### Data Filters")
    try:
        train_index = get_loader_index(train_path)
    except Exception as e:
        st.error(f"Could not read train dataset index: {e}")
        st.stop()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_classes = sorted(train_index.get("classes", []))
        sel_classes = st.multiselect("Classes", all_classes, default=[],
                                     key="feat_classes", help="Empty = all classes")
        all_loads = sorted(set(train_index.get("loads", [])))
        sel_loads = st.multiselect("Loads", all_loads, default=[],
                                   key="feat_loads", help="Empty = all loads")
    with col_f2:
        all_freqs = sorted(train_index.get("electrical_frequencies_hz", []))
        sel_freqs = st.multiselect("Frequencies (Hz)", all_freqs, default=[],
                                   key="feat_freqs", help="Empty = all frequencies")
        all_sensor_types = sorted(train_index.get("sensor_types", []))
        sel_sensor_types = st.multiselect(
            "Sensor types", all_sensor_types,
            default=["vibration"] if "vibration" in all_sensor_types else all_sensor_types[:1],
            key="feat_sensor_types",
        )

    _train_files = get_filtered_file_list(
        train_path,
        classes=tuple(sel_classes) if sel_classes else None,
        loads=tuple(float(v) for v in sel_loads) if sel_loads else None,
        frequencies=tuple(float(v) for v in sel_freqs) if sel_freqs else None,
        sensor_types=tuple(sel_sensor_types) if sel_sensor_types else None,
    )
    st.info(f"**{len(_train_files)}** files match current filters")

    # ── Windowing settings ────────────────────────────────────────────────────
    st.markdown("### Windowing")
    st.caption("Window size and overlap are configured in **2 · Preprocessing**.")
    window_size = int(st.session_state.get("win_size", 10000))
    overlap_ratio = float(st.session_state.get("win_overlap", 0.5))
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.metric("Window size (samples)", window_size)
        st.metric("Overlap ratio", overlap_ratio)
    with col_w2:
        shuffle = st.toggle("Shuffle windows", value=True, key="win_shuffle")
        random_state = st.number_input("Random state", min_value=0, max_value=9999,
                                       value=42, step=1, key="win_random")

    # ── Feature configuration ─────────────────────────────────────────────────
    st.markdown("### Feature Configuration")
    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        _default_sensor_idx = 0
        if sel_sensor_types and "current" in sel_sensor_types and "vibration" not in sel_sensor_types:
            _default_sensor_idx = 1
        sensor_type = st.selectbox("Sensor type", ["vibration", "current"],
                                   index=_default_sensor_idx, key="feat_sensor_type")
        _max_ch = 4 if sensor_type == "vibration" else 2
        _all_ch_keys = [f"ch{i + 1}" for i in range(_max_ch)]
        sel_channels = st.multiselect("Selected channels", _all_ch_keys,
                                      default=_all_ch_keys, key="feat_channels")
    with col_fc2:
        st.markdown("**Feature families**")
        enable_time = st.toggle("Time domain", value=True, key="feat_time")
        enable_freq = st.toggle("Frequency domain", value=(sensor_type == "vibration"),
                                key="feat_freq")
        enable_hilbert = st.toggle("Hilbert envelope", value=(sensor_type == "current"),
                                   key="feat_hilbert")

    if st.button("Extract Train Features", type="primary", key="btn_extract_train",
                 disabled=len(_train_files) == 0):
        from ml_toolbox.data_loader import FeatureConfig
        feat_conf = FeatureConfig(
            sensor_type=sensor_type,
            selected_channels=sel_channels if sel_channels else None,
        )
        if not enable_time:
            feat_conf.disable("time_domain")
        if not enable_freq:
            feat_conf.disable("frequency_domain")
        if not enable_hilbert:
            feat_conf.disable("hilbert_envelope")

        _progress_bar = st.progress(0)
        _status_text = st.empty()
        try:
            features, labels, feature_names, win_metadata, label_map = run_lazy_extraction(
                dataset_path=train_path,
                file_list=_train_files,
                lpf_enabled=st.session_state["lpf_enabled"],
                lpf_cutoff=float(st.session_state["lpf_cutoff"]),
                lpf_order=int(st.session_state["lpf_order"]),
                window_size=int(window_size),
                overlap_ratio=float(overlap_ratio),
                shuffle=shuffle,
                random_state=int(random_state),
                sensor_type=sensor_type,
                feature_config=feat_conf,
                class_to_int=None,
                progress_bar=_progress_bar,
                status_text=_status_text,
            )
            _progress_bar.progress(1.0)
            _status_text.empty()
            st.session_state[S.FEATURES] = features
            st.session_state[S.LABELS] = labels
            st.session_state[S.FEATURE_NAMES] = feature_names
            st.session_state[S.LABEL_MAP] = label_map
            st.session_state[S.DATASET_PATH] = train_path
            for _k in (S.PIPELINE, S.PREDICTIONS, S.SHAP_VALUES,
                       S.TEST_FEATURES, S.TEST_FEATURE_NAMES, S.TEST_LABELS):
                st.session_state.pop(_k, None)
            st.rerun()
        except Exception as e:
            st.error(f"Extraction failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if S.FEATURES in st.session_state:
        features = st.session_state[S.FEATURES]
        feature_names = st.session_state[S.FEATURE_NAMES]
        labels = st.session_state[S.LABELS]
        label_map = st.session_state[S.LABEL_MAP]

        ram_mb = features.nbytes / 1024 ** 2
        st.success(f"Features: {features.shape}  —  {len(feature_names)} features  —  {ram_mb:.1f} MB")

        counts = pd.Series(labels).value_counts().sort_index()
        counts.index = [label_map.get(i, str(i)) for i in counts.index]
        fig_wpc, ax = plt.subplots(figsize=(6, 3))
        ax.bar(counts.index, counts.values, color="#4CAF50")
        ax.set_xlabel("Class")
        ax.set_ylabel("Windows")
        ax.set_title("Windows per class")
        plt.tight_layout()
        st.pyplot(fig_wpc, width="content")
        plt.close(fig_wpc)

        with st.expander("Feature statistics", expanded=False):
            stats_df = pd.DataFrame(features, columns=feature_names).describe().T
            st.dataframe(stats_df, width='stretch')

        st.markdown("**Correlation heatmap**")
        fig_corr = correlation_heatmap(features, feature_names)
        st.pyplot(fig_corr, width="stretch")
        plt.close(fig_corr)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Train & Evaluate
# ═════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train & Evaluate")

    if S.FEATURES not in st.session_state:
        st.info("Extract train features in **3 · Feature Extraction** first.")
    else:
        st.markdown("### Test Dataset")

        try:
            test_index = get_loader_index(test_path)
        except Exception as e:
            st.error(f"Could not read test dataset index: {e}")
            test_index = {}

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            all_test_classes = sorted(test_index.get("classes", []))
            sel_test_classes = st.multiselect("Classes", all_test_classes, default=[],
                                              key="test_classes", help="Empty = all")
            all_test_loads = sorted(set(test_index.get("loads", [])))
            sel_test_loads = st.multiselect("Loads", all_test_loads, default=[],
                                            key="test_loads", help="Empty = all")
        with col_t2:
            all_test_freqs = sorted(test_index.get("electrical_frequencies_hz", []))
            sel_test_freqs = st.multiselect("Frequencies (Hz)", all_test_freqs, default=[],
                                            key="test_freqs", help="Empty = all")
            all_test_sensors = sorted(test_index.get("sensor_types", []))
            sel_test_sensors = st.multiselect(
                "Sensor types", all_test_sensors,
                default=["vibration"] if "vibration" in all_test_sensors else all_test_sensors[:1],
                key="test_sensor_types",
            )

        _test_files = get_filtered_file_list(
            test_path,
            classes=tuple(sel_test_classes) if sel_test_classes else None,
            loads=tuple(float(v) for v in sel_test_loads) if sel_test_loads else None,
            frequencies=tuple(float(v) for v in sel_test_freqs) if sel_test_freqs else None,
            sensor_types=tuple(sel_test_sensors) if sel_test_sensors else None,
        )
        st.info(f"**{len(_test_files)}** test files match current filters")

        if st.button("Extract Test Features", type="primary", key="btn_test_extract",
                     disabled=len(_test_files) == 0):
            from ml_toolbox.data_loader import FeatureConfig

            _train_label_map = st.session_state[S.LABEL_MAP]
            _class_to_int = {v: k for k, v in _train_label_map.items()}
            _test_sensor = st.session_state.get("feat_sensor_type", "vibration")
            _test_channels = st.session_state.get("feat_channels") or None

            _fc = FeatureConfig(sensor_type=_test_sensor, selected_channels=_test_channels)
            if not st.session_state.get("feat_time", True):
                _fc.disable("time_domain")
            if not st.session_state.get("feat_freq", True):
                _fc.disable("frequency_domain")
            if not st.session_state.get("feat_hilbert", False):
                _fc.disable("hilbert_envelope")

            _pb = st.progress(0)
            _st_txt = st.empty()
            try:
                _tf, _tl, _tfn, _, _ = run_lazy_extraction(
                    dataset_path=test_path,
                    file_list=_test_files,
                    lpf_enabled=st.session_state["lpf_enabled"],
                    lpf_cutoff=float(st.session_state["lpf_cutoff"]),
                    lpf_order=int(st.session_state["lpf_order"]),
                    window_size=int(st.session_state.get("win_size", 10000)),
                    overlap_ratio=float(st.session_state.get("win_overlap", 0.5)),
                    shuffle=False,
                    random_state=int(st.session_state.get("win_random", 42)),
                    sensor_type=_test_sensor,
                    feature_config=_fc,
                    class_to_int=_class_to_int,
                    progress_bar=_pb,
                    status_text=_st_txt,
                )
                _pb.progress(1.0)
                _st_txt.empty()
                st.session_state[S.TEST_FEATURES] = _tf
                st.session_state[S.TEST_LABELS] = _tl
                st.session_state[S.TEST_FEATURE_NAMES] = _tfn
                st.session_state[S.TEST_DATASET_PATH] = test_path
                st.session_state.pop(S.PIPELINE, None)
                st.session_state.pop(S.PREDICTIONS, None)
                st.session_state.pop(S.SHAP_VALUES, None)
                st.rerun()
            except Exception as e:
                st.error(f"Test feature extraction failed: {e}")

        if S.TEST_FEATURES in st.session_state:
            st.success(f"Test features: {st.session_state[S.TEST_FEATURES].shape}  (float32)")

        # ── RF hyperparameters ────────────────────────────────────────────────
        st.markdown("### Model")
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            n_estimators = st.slider("n_estimators", 10, 500, 100, 10, key="rf_n_est")
            max_depth = st.slider("max_depth", 1, 50, 10, 1, key="rf_max_depth")
        with col_rf2:
            min_samples_split = st.slider("min_samples_split", 2, 20, 5, 1, key="rf_min_split")
            min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 2, 1, key="rf_min_leaf")

        _train_ready = S.TEST_FEATURES in st.session_state
        if not _train_ready:
            st.info("Extract test features above, then train.")

        if _train_ready and st.button("Train & Evaluate", type="primary", key="btn_train"):
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            try:
                with st.spinner("Training…"):
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("rf", RandomForestClassifier(
                            n_estimators=int(n_estimators),
                            random_state=42,
                            max_depth=int(max_depth),
                            min_samples_split=int(min_samples_split),
                            min_samples_leaf=int(min_samples_leaf),
                        )),
                    ])
                    pipe.fit(st.session_state[S.FEATURES], st.session_state[S.LABELS])
                    preds = pipe.predict(st.session_state[S.TEST_FEATURES])
                    st.session_state[S.PIPELINE] = pipe
                    st.session_state[S.PREDICTIONS] = preds
            except Exception as e:
                st.error(f"Training failed: {e}")

        if S.PIPELINE in st.session_state and S.PREDICTIONS in st.session_state:
            from sklearn.metrics import f1_score, classification_report, confusion_matrix

            pipe = st.session_state[S.PIPELINE]
            preds = st.session_state[S.PREDICTIONS]
            test_labels = st.session_state[S.TEST_LABELS]
            test_features_ev = st.session_state[S.TEST_FEATURES]

            accuracy = pipe.score(test_features_ev, test_labels)
            f1 = f1_score(test_labels, preds, average="macro")

            mc1, mc2 = st.columns(2)
            mc1.metric("Test Accuracy", f"{accuracy:.4f}")
            mc2.metric("F1 (Macro)", f"{f1:.4f}")

            label_map = st.session_state.get(S.LABEL_MAP, {})
            class_names_str = [label_map.get(c, str(c)) for c in pipe.classes_]
            cm = confusion_matrix(test_labels, preds, labels=pipe.classes_)
            fig_cm = confusion_matrix_chart(cm, class_names_str)
            st.pyplot(fig_cm, width="content")
            plt.close(fig_cm)

            with st.expander("Classification Report", expanded=True):
                st.text(classification_report(test_labels, preds))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — SHAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.subheader("SHAP Analysis")

    if S.PIPELINE not in st.session_state:
        st.info("Train a model in the **Train & Evaluate** tab first.")
    elif S.TEST_FEATURES not in st.session_state:
        st.info("Test features are missing — process the test set in the Train & Evaluate tab.")
    else:
        if st.button("Run SHAP", type="primary", key="btn_shap"):
            import shap
            try:
                with st.spinner("Computing SHAP values (this may take a minute)…"):
                    pipe = st.session_state[S.PIPELINE]
                    scaler = pipe.named_steps["scaler"]
                    rf = pipe.named_steps["rf"]
                    test_features_scaled = scaler.transform(st.session_state[S.TEST_FEATURES])
                    explainer = shap.TreeExplainer(
                        rf,
                        feature_perturbation="tree_path_dependent",
                        model_output="raw",
                    )
                    shap_values = explainer.shap_values(test_features_scaled)
                st.session_state[S.SHAP_VALUES] = shap_values
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

        if S.SHAP_VALUES in st.session_state:
            import shap

            shap_values = st.session_state[S.SHAP_VALUES]
            test_features = st.session_state[S.TEST_FEATURES]
            feat_names = st.session_state[S.TEST_FEATURE_NAMES]
            pipe = st.session_state[S.PIPELINE]
            scaler = pipe.named_steps["scaler"]
            test_features_scaled = scaler.transform(test_features)

            if isinstance(shap_values, list):
                shap_arr = np.stack(shap_values, axis=-1)
            else:
                shap_arr = shap_values

            n_classes = shap_arr.shape[-1] if shap_arr.ndim == 3 else 1
            label_map = st.session_state.get(S.LABEL_MAP, {c: str(c) for c in range(n_classes)})

            class_options = {label_map.get(i, str(i)): i for i in range(n_classes)}
            selected_class_name = st.selectbox("Class", list(class_options.keys()), key="shap_class")
            class_idx = class_options[selected_class_name]

            shap_for_class = shap_arr[:, :, class_idx] if shap_arr.ndim == 3 else shap_arr

            mean_abs = np.abs(shap_for_class).mean(axis=0)
            top_n = st.slider("Top N features", 5, len(feat_names), min(15, len(feat_names)),
                              key="shap_topn")
            top_idx = np.argsort(mean_abs)[::-1][:top_n]
            top_df = pd.DataFrame({
                "feature": [feat_names[i] for i in top_idx],
                "mean |SHAP|": mean_abs[top_idx],
            })

            fig_bar, ax_bar = plt.subplots(figsize=(8, max(3, top_n * 0.35)))
            ax_bar.barh(top_df["feature"][::-1], top_df["mean |SHAP|"][::-1], color="#2196F3")
            ax_bar.set_xlabel("Mean |SHAP value|")
            ax_bar.set_title(f"Top {top_n} features — class '{selected_class_name}'")
            plt.tight_layout()
            st.pyplot(fig_bar, width="content")
            plt.close(fig_bar)

            st.dataframe(top_df, width='stretch')

            with st.expander("Beeswarm plot", expanded=False):
                exp = shap.Explanation(
                    values=shap_for_class,
                    data=test_features_scaled,
                    feature_names=feat_names,
                )
                fig_bee, _ = plt.subplots()
                shap.plots.beeswarm(exp, show=False)
                st.pyplot(fig_bee, width="stretch")
                plt.close(fig_bee)

            with st.expander("SHAP heatmap", expanded=False):
                exp_heat = shap.Explanation(
                    values=shap_for_class,
                    data=test_features_scaled,
                    feature_names=feat_names,
                )
                fig_heat, _ = plt.subplots()
                shap.plots.heatmap(exp_heat, show=False)
                st.pyplot(fig_heat, width="stretch")
                plt.close(fig_heat)
