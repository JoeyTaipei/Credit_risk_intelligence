"""SHAP visualization helpers for model explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def _is_tree_model(model: Any) -> bool:
    """Return whether the model looks like an XGBoost or sklearn tree model."""
    module_name = model.__class__.__module__.lower()
    class_name = model.__class__.__name__.lower()
    tree_markers = ("xgboost", "forest", "tree", "gradientboosting", "histgradientboosting")
    return any(marker in module_name or marker in class_name for marker in tree_markers)


def plot_shap_summary(
    model: Any,
    X: Any,
    feature_names: Sequence[str],
    save_path: str,
) -> Any:
    """Create and save a SHAP summary plot for a trained model.

    Args:
        model: Trained model object used to compute SHAP values.
        X: Feature matrix as a numpy array or pandas DataFrame.
        feature_names: Feature names matching the columns in X.
        save_path: Destination path for the PNG image.

    Returns:
        Computed SHAP values for downstream analysis.
    """
    import matplotlib.pyplot as plt
    import shap

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TreeExplainer is faster and more stable for XGBoost and sklearn tree ensembles.
    explainer = shap.TreeExplainer(model) if _is_tree_model(model) else shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=list(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()
    return shap_values


def plot_shap_waterfall(
    shap_values: Any,
    X_sample: Any,
    feature_names: Sequence[str],
    save_path: str,
) -> None:
    """Create and save a SHAP waterfall plot for one prediction.

    Args:
        shap_values: Precomputed SHAP values from an explainer.
        X_sample: Single feature row represented as an array, Series, or DataFrame row.
        feature_names: Feature names matching X_sample.
        save_path: Destination path for the PNG image.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import shap

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Newer SHAP returns Explanation objects; older paths can still pass raw arrays.
    sample_values = shap_values[0] if hasattr(shap_values, "__getitem__") else shap_values
    if hasattr(sample_values, "feature_names"):
        sample_values.feature_names = list(feature_names)
    if hasattr(sample_values, "data") and sample_values.data is None:
        sample_values.data = X_sample

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(sample_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()
