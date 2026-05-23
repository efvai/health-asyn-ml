"""Tab 5 — SHAP Analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import state as S


def render() -> None:
    st.subheader("SHAP Analysis")

    if S.PIPELINE not in st.session_state:
        st.info("Train a model in the **Train & Evaluate** tab first.")
        return
    if S.TEST_FEATURES not in st.session_state:
        st.info("Test features are missing — process the test set in the Train & Evaluate tab.")
        return

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
