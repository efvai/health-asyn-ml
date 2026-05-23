"""Tab 4 — Train & Evaluate."""

import matplotlib.pyplot as plt
import streamlit as st

import state as S
from ml_toolbox.data_loader import FeatureConfig, TIME_FEATURES, FREQ_FEATURES
from utils import (
    get_loader_index,
    get_filtered_file_list,
    run_lazy_extraction,
    confusion_matrix_chart,
)


def render(train_path: str, test_path: str) -> None:
    st.subheader("Train & Evaluate")

    if S.FEATURES not in st.session_state:
        st.info("Extract train features in **3 · Feature Extraction** first.")
        return

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
        _train_label_map = st.session_state[S.LABEL_MAP]
        _class_to_int = {v: k for k, v in _train_label_map.items()}
        _test_sensor = st.session_state.get("feat_sensor_type", "vibration")
        _t_max_ch = 4 if _test_sensor == "vibration" else 2
        _t_all_ch_keys = [f"ch{i + 1}" for i in range(_t_max_ch)]
        _t_channels = [_c for _c in _t_all_ch_keys if st.session_state.get(f"fc_{_c}", True)]
        _t_td = [_f for _f in TIME_FEATURES if st.session_state.get(f"fc_td_{_f}", True)]
        _t_fd = [_f for _f in FREQ_FEATURES if st.session_state.get(f"fc_fd_{_f}", True)]
        _sel_features = [f"{ch}_{feat}" for ch in _t_channels for feat in (_t_td + _t_fd)]

        _fc = FeatureConfig(features=_sel_features) if _sel_features else None
        if _fc is None:
            st.error("No features selected — configure features in 3 · Feature Extraction first.")
            st.stop()

        _pb = st.progress(0)
        _st_txt = st.empty()
        try:
            _tf, _tl, _tfn, _, _ = run_lazy_extraction(
                dataset_path=test_path,
                file_list=_test_files,
                lpf_enabled=st.session_state["lpf_enabled"],
                lpf_cutoff=float(st.session_state["lpf_cutoff"]),
                lpf_order=int(st.session_state["lpf_order"]),
                detrend_enabled=st.session_state.get("detrend_enabled", False),
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

    # ── RF hyperparameters ────────────────────────────────────────────────────
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
