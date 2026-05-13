"""Optimize a weighted OOF ensemble across XGBoost, LightGBM, and CatBoost."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from src.data.features_v2 import TARGET_COL, load_credit_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training2.csv"
OOF_FILES = {
    "xgb": PROCESSED_DIR / "xgb_oof.npy",
    "lgbm": PROCESSED_DIR / "lgbm_oof.npy",
    "catboost": PROCESSED_DIR / "catboost_oof.npy",
}


def load_target(data_path: Path | str = DEFAULT_DATA_PATH) -> np.ndarray:
    """Load the v2 target vector in the same row order used by OOF arrays."""
    feature_df = load_credit_features_v2(data_path)
    return feature_df[TARGET_COL].astype(int).to_numpy()


def load_oof_predictions() -> dict[str, np.ndarray]:
    """Load saved OOF predictions for all base models."""
    missing = [str(path) for path in OOF_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing OOF predictions. Run train_xgb_v2.py and "
            f"train_lgbm_catboost.py first. Missing: {missing}"
        )
    return {model_name: np.load(path) for model_name, path in OOF_FILES.items()}


def _weighted_average(oofs: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.column_stack([oofs["xgb"], oofs["lgbm"], oofs["catboost"]])
    return stacked @ weights


def grid_search_weights(
    y: np.ndarray,
    oofs: dict[str, np.ndarray],
    *,
    step: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Search positive weights on a coarse 0.1 grid."""
    best_weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    best_auc = -np.inf
    values = np.round(np.arange(step, 1.0, step), 10)

    for weights_tuple in product(values, repeat=3):
        weights = np.asarray(weights_tuple, dtype=np.float64)
        if not np.isclose(weights.sum(), 1.0):
            continue
        ensemble_oof = _weighted_average(oofs, weights)
        auc = float(roc_auc_score(y, ensemble_oof))
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    return best_weights, best_auc


def optimize_weights(
    y: np.ndarray,
    oofs: dict[str, np.ndarray],
    initial_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fine-tune ensemble weights with constrained SciPy optimization."""

    def objective(weights: np.ndarray) -> float:
        ensemble_oof = _weighted_average(oofs, weights)
        return -float(roc_auc_score(y, ensemble_oof))

    result = minimize(
        objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        print(f"[WARN] Weight optimizer did not fully converge: {result.message}")
        weights = initial_weights
    else:
        weights = np.asarray(result.x, dtype=np.float64)

    weights = weights / weights.sum()
    auc = float(roc_auc_score(y, _weighted_average(oofs, weights)))
    return weights, auc


def run_ensemble(
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> dict[str, float | dict[str, float] | str]:
    """Load OOF arrays, optimize weights, print report, and save weights."""
    y = load_target(data_path)
    oofs = load_oof_predictions()
    for model_name, values in oofs.items():
        if len(values) != len(y):
            raise ValueError(
                f"{model_name} OOF length {len(values)} does not match target length {len(y)}"
            )

    individual_aucs = {
        "xgb": float(roc_auc_score(y, oofs["xgb"])),
        "lgbm": float(roc_auc_score(y, oofs["lgbm"])),
        "catboost": float(roc_auc_score(y, oofs["catboost"])),
    }

    grid_weights, grid_auc = grid_search_weights(y, oofs)
    best_weights, ensemble_auc = optimize_weights(y, oofs, grid_weights)

    weight_map = {
        "xgb": float(best_weights[0]),
        "lgbm": float(best_weights[1]),
        "catboost": float(best_weights[2]),
    }
    result = {
        "data_path": str(data_path),
        "individual_auc": individual_aucs,
        "grid_auc": float(grid_auc),
        "ensemble_auc": float(ensemble_auc),
        "weights": weight_map,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "ensemble_weights.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Individual OOF AUCs:")
    print(f"Labels: {data_path}")
    print(f"- XGBoost:  {individual_aucs['xgb']:.6f}")
    print(f"- LightGBM: {individual_aucs['lgbm']:.6f}")
    print(f"- CatBoost: {individual_aucs['catboost']:.6f}")
    print(
        "Best ensemble weights: "
        f"XGBoost={weight_map['xgb']:.4f}, "
        f"LightGBM={weight_map['lgbm']:.4f}, "
        f"CatBoost={weight_map['catboost']:.4f}"
    )
    print(f"Ensemble OOF AUC: {ensemble_auc:.6f}")

    print("\nModel               | OOF AUC | Notes")
    print("--------------------|---------|------------------")
    print("XGBoost (original)  | 0.6700  | single split, default params")
    print(f"XGBoost (tuned)     | {individual_aucs['xgb']:.4f}  | Optuna + 5-fold CV")
    print(f"LightGBM            | {individual_aucs['lgbm']:.4f}  | Optuna + 5-fold CV")
    print(f"CatBoost            | {individual_aucs['catboost']:.4f}  | default + 5-fold CV")
    print(f"Ensemble (weighted) | {ensemble_auc:.4f}  | optimized weights")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV path for labels. Default: data/raw/cs-training2.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ensemble(data_path=args.data_path)


if __name__ == "__main__":
    main()
