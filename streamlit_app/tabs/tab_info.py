"""Tab 1 — Dataset Info."""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils import get_loader_index


def render(train_path: str, test_path: str) -> None:
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
