"""Session state key constants — use these everywhere instead of raw strings."""

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASET_PATH    = "dataset_path"     # str active train dataset path
TEST_DATASET_PATH = "test_dataset_path"

# ── Train features ────────────────────────────────────────────────────────────
FEATURES        = "features"         # np.ndarray (n, n_features) float32
FEATURE_NAMES   = "feature_names"    # List[str]
LABELS          = "labels"           # np.ndarray (n,) int32
LABEL_MAP       = "label_map"        # Dict[int, str] label id -> class name

# ── Test features ─────────────────────────────────────────────────────────────
TEST_FEATURES      = "test_features"
TEST_FEATURE_NAMES = "test_feature_names"
TEST_LABELS        = "test_labels"

# ── Model ─────────────────────────────────────────────────────────────────────
PIPELINE        = "pipeline"         # sklearn Pipeline (scaler + RF)
PREDICTIONS     = "predictions"      # np.ndarray test predictions

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_VALUES     = "shap_values"      # np.ndarray (n, n_feat, n_classes)
