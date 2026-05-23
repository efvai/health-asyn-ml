"""Tab 3 — Feature Extraction."""

import io
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import state as S
from ml_toolbox.data_loader import FeatureConfig, TIME_FEATURES, FREQ_FEATURES
from utils import (
    get_loader_index,
    get_filtered_file_list,
    run_lazy_extraction,
    correlation_heatmap,
)


def render(train_path: str) -> None:
    st.subheader("Feature Extraction")

    _lpf_status = (
        f"LPF ON — cutoff {st.session_state['lpf_cutoff']} Hz, "
        f"order {st.session_state['lpf_order']}"
        if st.session_state["lpf_enabled"] else "LPF OFF"
    )
    _detrend_status = "Detrend ON" if st.session_state.get("detrend_enabled", False) else "Detrend OFF"
    st.caption(f"{_lpf_status} · {_detrend_status} · Configure in **2 · Preprocessing**")

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

    _default_sensor_idx = 0
    if sel_sensor_types and "current" in sel_sensor_types and "vibration" not in sel_sensor_types:
        _default_sensor_idx = 1
    sensor_type = st.selectbox("Sensor type", ["vibration", "current"],
                               index=_default_sensor_idx, key="feat_sensor_type")

    _max_ch = 4 if sensor_type == "vibration" else 2
    _all_ch_keys = [f"ch{i + 1}" for i in range(_max_ch)]

    # Initialize session state defaults (only if not already set)
    for _c in _all_ch_keys:
        if f"fc_{_c}" not in st.session_state:
            st.session_state[f"fc_{_c}"] = True
    for _f in TIME_FEATURES:
        if f"fc_td_{_f}" not in st.session_state:
            st.session_state[f"fc_td_{_f}"] = True
    for _f in FREQ_FEATURES:
        if f"fc_fd_{_f}" not in st.session_state:
            st.session_state[f"fc_fd_{_f}"] = True

    _col_ch, _col_td, _col_fd = st.columns(3)
    with _col_ch:
        st.markdown("**Channels**")
        _btn_ch_all, _btn_ch_none = st.columns(2)
        if _btn_ch_all.button("All", key="fc_ch_all"):
            for _c in _all_ch_keys:
                st.session_state[f"fc_{_c}"] = True
            st.rerun()
        if _btn_ch_none.button("None", key="fc_ch_none"):
            for _c in _all_ch_keys:
                st.session_state[f"fc_{_c}"] = False
            st.rerun()
        for _c in _all_ch_keys:
            st.checkbox(_c, key=f"fc_{_c}")

    with _col_td:
        st.markdown("**Time-domain**")
        _btn_td_all, _btn_td_none = st.columns(2)
        if _btn_td_all.button("All", key="fc_td_all"):
            for _f in TIME_FEATURES:
                st.session_state[f"fc_td_{_f}"] = True
            st.rerun()
        if _btn_td_none.button("None", key="fc_td_none"):
            for _f in TIME_FEATURES:
                st.session_state[f"fc_td_{_f}"] = False
            st.rerun()
        for _f in TIME_FEATURES:
            st.checkbox(_f, key=f"fc_td_{_f}")

    with _col_fd:
        st.markdown("**Frequency-domain**")
        _btn_fd_all, _btn_fd_none = st.columns(2)
        if _btn_fd_all.button("All", key="fc_fd_all"):
            for _f in FREQ_FEATURES:
                st.session_state[f"fc_fd_{_f}"] = True
            st.rerun()
        if _btn_fd_none.button("None", key="fc_fd_none"):
            for _f in FREQ_FEATURES:
                st.session_state[f"fc_fd_{_f}"] = False
            st.rerun()
        for _f in FREQ_FEATURES:
            st.checkbox(_f, key=f"fc_fd_{_f}")

    _sel_channels = [_c for _c in _all_ch_keys if st.session_state.get(f"fc_{_c}", True)]
    _sel_td = [_f for _f in TIME_FEATURES if st.session_state.get(f"fc_td_{_f}", True)]
    _sel_fd = [_f for _f in FREQ_FEATURES if st.session_state.get(f"fc_fd_{_f}", True)]
    _sel_feat_names = _sel_td + _sel_fd
    sel_features = [f"{ch}_{feat}" for ch in _sel_channels for feat in _sel_feat_names]
    st.caption(
        f"**{len(_sel_channels)}** channels \u00d7 **{len(_sel_feat_names)}** features "
        f"= **{len(sel_features)}** feature columns"
    )

    if st.button("Extract Train Features", type="primary", key="btn_extract_train",
                 disabled=not sel_features or len(_train_files) == 0):
        feat_conf = FeatureConfig(features=sel_features)

        _progress_bar = st.progress(0)
        _status_text = st.empty()
        try:
            features, labels, feature_names, win_metadata, label_map = run_lazy_extraction(
                dataset_path=train_path,
                file_list=_train_files,
                lpf_enabled=st.session_state["lpf_enabled"],
                lpf_cutoff=float(st.session_state["lpf_cutoff"]),
                lpf_order=int(st.session_state["lpf_order"]),
                detrend_enabled=st.session_state.get("detrend_enabled", False),
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
            st.session_state[S.WIN_METADATA] = win_metadata
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
        win_metadata = st.session_state.get(S.WIN_METADATA, [])

        ram_mb = features.nbytes / 1024 ** 2
        st.success(f"Features: {features.shape}  —  {len(feature_names)} features  —  {ram_mb:.1f} MB")

        def _build_excel_bytes() -> bytes:
            df_exp = pd.DataFrame(features, columns=feature_names)
            mapped = [label_map.get(i, str(i)) for i in labels]
            df_exp.insert(0, "class", mapped)
            # Prepend window location columns so rows can be cross-referenced
            # with the Before/After preview in the Preprocessing tab
            _meta_cols = ["sample_id", "window_id", "start_sample", "end_sample"]
            for _col_idx, _key in enumerate(_meta_cols):
                _vals = [m.get(_key, "") for m in win_metadata] if win_metadata else [""] * len(df_exp)
                df_exp.insert(_col_idx, _key, _vals)
            buf = io.BytesIO()
            df_exp.to_excel(buf, index=False, engine="openpyxl")
            return buf.getvalue()

        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Export feature matrix (.xlsx)",
            data=_build_excel_bytes(),
            file_name=f"features_{_ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="btn_export_features",
        )

        with st.expander("Feature statistics", expanded=False):
            df_all = pd.DataFrame(features, columns=feature_names)
            mapped_labels = [label_map.get(i, str(i)) for i in labels]
            df_all["class"] = mapped_labels
            
            grouped_stats = df_all.groupby("class").describe()
            
            stats_df = grouped_stats.T.unstack(level=1)
            st.dataframe(stats_df, width='stretch')

        with st.expander("Correlation heatmap", expanded=False):
            fig_corr = correlation_heatmap(features, feature_names)
            st.pyplot(fig_corr, width="stretch")
            plt.close(fig_corr)
