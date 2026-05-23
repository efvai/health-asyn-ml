"""
Streamlit ML Toolbox Pipeline Debugger
Run: streamlit run streamlit_app/app.py
"""

import sys
import streamlit as st

# Ensure project root is on path before any ml_toolbox import
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    get_project_root,
    discover_datasets,
    HF_REPO_ID,
    list_hf_datasets,
    download_hf_dataset,
)
from tabs import tab_info, tab_prep, tab_features, tab_train, tab_shap

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
    ("detrend_enabled", False),
    ("win_size", 10000), ("win_overlap", 0.5),
    # Peak finder defaults
    ("peak_enabled", False),
    ("peak_prominence", 0.02),
    ("peak_distance_hz", 2.0),
    ("peak_n_harmonics", 5),
    ("peak_dom_freq_min", 10.0),
    ("peak_dom_freq_max", 60.0),
    ("peak_tolerance_hz", 2.0),
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
        index=AVAILABLE_DATASETS.index("data_set_4_2khz_float32") if "data_set_4_2khz_float32" in AVAILABLE_DATASETS else 0,
        key="sidebar_train_dataset",
    )
    test_dataset = st.selectbox(
        "Test dataset",
        AVAILABLE_DATASETS,
        index=AVAILABLE_DATASETS.index("data_set_6_2khz_float32") if "data_set_6_2khz_float32" in AVAILABLE_DATASETS else 0,
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
_tab_info, _tab_prep, _tab_features, _tab_train, _tab_shap = st.tabs([
    "1 · Dataset Info",
    "2 · Preprocessing",
    "3 · Feature Extraction",
    "4 · Train & Evaluate",
    "5 · SHAP",
])

with _tab_info:
    tab_info.render(train_path, test_path)


with _tab_prep:
    tab_prep.render(train_path)


with _tab_features:
    tab_features.render(train_path)


with _tab_train:
    tab_train.render(train_path, test_path)


with _tab_shap:
    tab_shap.render()
