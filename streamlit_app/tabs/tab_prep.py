"""Tab 2 — Preprocessing."""

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    get_loader_index,
    cached_load_single_raw,
    time_domain_chart,
    window_frequency_chart,
)


def render(train_path: str) -> None:
    st.subheader("Preprocessing")

    st.markdown("### Detrending")
    st.toggle("Enable Detrending", key="detrend_enabled")
    st.caption("Removes linear trend from each signal. Applied before LPF.")

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

                from ml_toolbox import ButterworthLPF, DetrendingFilter, PreprocessorPipeline
                _steps = []
                if st.session_state.get("detrend_enabled", False):
                    _steps.append(DetrendingFilter())
                if st.session_state["lpf_enabled"]:
                    _steps.append(ButterworthLPF(
                        cutoff_hz=float(st.session_state["lpf_cutoff"]),
                        order=int(st.session_state["lpf_order"]),
                    ))
                _pipeline = PreprocessorPipeline(steps=_steps)
                try:
                    filtered = _pipeline.apply(raw, fs=fs)
                except Exception as _e:
                    st.error(f"Preprocessing error: {_e}")
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
                    _peak_settings = {
                        "enabled": st.session_state["peak_enabled"],
                        "prominence": st.session_state["peak_prominence"],
                        "distance_hz": st.session_state["peak_distance_hz"],
                        "n_harmonics": st.session_state["peak_n_harmonics"],
                        "dom_freq_min": st.session_state["peak_dom_freq_min"],
                        "dom_freq_max": st.session_state["peak_dom_freq_max"],
                        "tolerance_hz": st.session_state["peak_tolerance_hz"],
                    }
                    _chart_result = window_frequency_chart(
                        raw_win, filt_win, fs, ch_sel,
                        freq_mode=freq_mode, nperseg=int(nperseg),
                        peak_settings=_peak_settings,
                    )
                    with st.expander("Peak Finder Settings", expanded=st.session_state["peak_enabled"]):
                        st.toggle("Enable peak overlay", key="peak_enabled")
                        _pk_disabled = not st.session_state["peak_enabled"]
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            st.slider(
                                "Prominence (fraction of max)", 0.001, 0.5, step=0.001,
                                key="peak_prominence", disabled=_pk_disabled,
                                help="Peaks must exceed this fraction of the spectrum maximum.",
                            )
                            st.slider(
                                "Min peak distance (Hz)", 0.5, 50.0, step=0.5,
                                key="peak_distance_hz", disabled=_pk_disabled,
                            )
                            st.number_input(
                                "Harmonics to show", min_value=1, max_value=10,
                                step=1, key="peak_n_harmonics", disabled=_pk_disabled,
                            )
                        with pc2:
                            st.slider(
                                "Dominant freq search min (Hz)", 1.0, 200.0, step=1.0,
                                key="peak_dom_freq_min", disabled=_pk_disabled,
                            )
                            st.slider(
                                "Dominant freq search max (Hz)", 1.0, 500.0, step=1.0,
                                key="peak_dom_freq_max", disabled=_pk_disabled,
                            )
                            st.slider(
                                "Harmonic match tolerance (Hz)", 0.5, 20.0, step=0.5,
                                key="peak_tolerance_hz", disabled=_pk_disabled,
                            )
                    if st.session_state["peak_enabled"]:
                        fig_freq, _dom_freq, _harmonics = _chart_result
                        st.plotly_chart(fig_freq, width='stretch')
                        with st.expander("Peak Analysis", expanded=True):
                            if _dom_freq is not None:
                                st.metric("Dominant frequency (f₀)", f"{_dom_freq:.2f} Hz")
                            else:
                                st.warning("No dominant frequency found in the specified range.")
                            if _harmonics:
                                _hdf = pd.DataFrame(_harmonics)
                                _hdf.columns = ["k", "Expected Hz", "Actual Hz", "Amplitude", "Peak found"]
                                st.dataframe(_hdf.style.format({"Expected Hz": "{:.2f}", "Actual Hz": "{:.2f}", "Amplitude": "{:.4g}"}), width='stretch')
                    else:
                        st.plotly_chart(_chart_result, width='stretch')
        else:
            st.info("No samples found in index.")
    else:
        st.info("No sensor types found — check the train dataset in the sidebar.")
