"""Training entry points for Credit Risk Intelligence."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import clean_tabular, create_synthetic_time_series, engineer_tabular_features
from src.models.xgb_baseline import predict, train_xgb_baseline
from src.utils.shap_visualizer import plot_shap_summary, plot_shap_waterfall


TARGET_COL = "SeriousDlqin2yrs"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CREDIT_PATH = PROJECT_ROOT / "data/raw/cs-training.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "docs/figures"
SYNTHETIC_TEXT_PATH = PROJECT_ROOT / "data/synthetic/loan_descriptions.csv"


def _metrics(y_true: Any, y_prob: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics at the fixed 0.5 threshold."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 train, validation, and test splits."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[TARGET_COL],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[TARGET_COL],
    )
    return train_df, val_df, test_df


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist split DataFrames as parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)


def _feature_target(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target."""
    return split_df.drop(columns=[TARGET_COL]), split_df[TARGET_COL]


def _top_shap_features(shap_values: Any, feature_names: list[str]) -> list[str]:
    """Return the top three features by mean absolute SHAP value."""
    values = getattr(shap_values, "values", shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:3]
    return [feature_names[index] for index in top_indices]


def run_xgb_stage() -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Run the Day 1 XGBoost baseline training stage.

    Returns:
        Validation metrics, test metrics, and top three SHAP feature names.
    """
    if not RAW_CREDIT_PATH.exists():
        raise FileNotFoundError(f"Missing required raw dataset: {RAW_CREDIT_PATH}")

    raw_df = pd.read_csv(RAW_CREDIT_PATH)
    clean_df = clean_tabular(raw_df)
    feature_df = engineer_tabular_features(clean_df)
    train_df, val_df, test_df = _split_data(feature_df)
    _save_splits(train_df, val_df, test_df)

    X_train, y_train = _feature_target(train_df)
    X_val, y_val = _feature_target(val_df)
    X_test, y_test = _feature_target(test_df)

    model, val_metrics = train_xgb_baseline(X_train, y_train, X_val, y_val)
    test_prob = predict(model, X_test)
    test_metrics = _metrics(y_test, test_prob)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "xgb_baseline.pkl").open("wb") as file:
        pickle.dump(model, file)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = list(X_train.columns)
    shap_values = plot_shap_summary(
        model,
        X_test,
        feature_names,
        str(FIGURES_DIR / "shap_summary_day1.png"),
    )

    high_risk_index = int(np.argmax(test_prob))
    plot_shap_waterfall(
        shap_values[high_risk_index : high_risk_index + 1],
        X_test.iloc[high_risk_index],
        feature_names,
        str(FIGURES_DIR / "shap_waterfall_day1.png"),
    )

    top_features = _top_shap_features(shap_values, feature_names)
    print(f"[INFO] Val metrics: {val_metrics}")
    print(f"[INFO] Test metrics: {test_metrics}")
    print(f"[INFO] Top 3 SHAP features: {top_features}")
    return val_metrics, test_metrics, top_features


def run_lstm_stage() -> float:
    """Run the Day 2 LSTM encoder training stage.

    Returns:
        Final validation ROC-AUC.
    """
    import mlflow
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.lstm_encoder import LSTMEncoder

    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing processed splits: expected {train_path} and {val_path}"
        )

    torch.manual_seed(42)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = torch.FloatTensor(create_synthetic_time_series(train_df))
    y_train = torch.FloatTensor(train_df[TARGET_COL].to_numpy()).view(-1, 1)
    X_val = torch.FloatTensor(create_synthetic_time_series(val_df))
    y_val = torch.FloatTensor(val_df[TARGET_COL].to_numpy()).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    head = nn.Linear(32, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = encoder.state_dict()
    epochs_without_improvement = 0
    final_val_auc = 0.0

    with mlflow.start_run():
        mlflow.log_params(
            {
                "stage": "lstm",
                "batch_size": 256,
                "epochs": 20,
                "patience": 5,
                "learning_rate": 1e-3,
            }
        )

        for epoch in range(1, 21):
            encoder.train()
            head.train()
            train_loss_total = 0.0
            train_examples = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = head(encoder(batch_x))
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_size = batch_x.size(0)
                train_loss_total += loss.item() * batch_size
                train_examples += batch_size

            train_loss = train_loss_total / train_examples

            encoder.eval()
            head.eval()
            val_loss_total = 0.0
            val_examples = 0
            val_logits: list[torch.Tensor] = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = head(encoder(batch_x))
                    loss = criterion(logits, batch_y)
                    batch_size = batch_x.size(0)
                    val_loss_total += loss.item() * batch_size
                    val_examples += batch_size
                    val_logits.append(logits)

            val_loss = val_loss_total / val_examples
            val_prob = torch.sigmoid(torch.cat(val_logits)).cpu().numpy().ravel()
            final_val_auc = float(roc_auc_score(y_val.numpy().ravel(), val_prob))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_auc", final_val_auc, step=epoch)

            print(
                f"[INFO] epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={final_val_auc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 5:
                    print(f"[INFO] Early stopping at epoch {epoch}.")
                    break

    encoder.load_state_dict(best_state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), PROCESSED_DIR / "lstm_encoder.pt")
    print(f"[INFO] Final val AUC: {final_val_auc:.4f}")
    return final_val_auc


def run_gnn_stage() -> tuple[float, dict]:
    """Run the Day 3 GraphSAGE encoder training stage.

    Returns:
        Final validation ROC-AUC and graph statistics.
    """
    import mlflow
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    from src.data.graph_builder import build_borrower_graph, get_graph_stats
    from src.models.gnn_encoder import GraphSAGEEncoder

    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing processed training split: {train_path}")

    torch.manual_seed(42)
    train_df = pd.read_parquet(train_path).reset_index(drop=True)
    edge_index, node_features = build_borrower_graph(train_df)
    labels = torch.FloatTensor(train_df[TARGET_COL].to_numpy()).view(-1, 1)
    stats = get_graph_stats(edge_index, n_nodes=node_features.shape[0])
    print(f"[INFO] Graph stats: {stats}")

    indices = np.arange(len(train_df))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=train_df[TARGET_COL],
    )
    val_indices, _ = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=42,
        stratify=train_df.iloc[temp_indices][TARGET_COL],
    )
    train_mask = torch.zeros(len(train_df), dtype=torch.bool)
    val_mask = torch.zeros(len(train_df), dtype=torch.bool)
    train_mask[torch.LongTensor(train_indices)] = True
    val_mask[torch.LongTensor(val_indices)] = True

    encoder = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    head = nn.Linear(32, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=5e-3,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = encoder.state_dict()
    epochs_without_improvement = 0
    final_val_auc = 0.0

    with mlflow.start_run():
        mlflow.log_params(
            {
                "stage": "gnn",
                "epochs": 50,
                "patience": 10,
                "learning_rate": 5e-3,
                "input_dim": 5,
                "embedding_dim": 32,
            }
        )
        mlflow.log_metrics(
            {
                "num_nodes": float(stats["num_nodes"]),
                "num_edges": float(stats["num_edges"]),
                "avg_degree": float(stats["avg_degree"]),
            }
        )

        for epoch in range(1, 51):
            encoder.train()
            head.train()
            optimizer.zero_grad()
            logits = head(encoder(node_features, edge_index))
            train_loss = criterion(logits[train_mask], labels[train_mask])
            train_loss.backward()
            optimizer.step()

            encoder.eval()
            head.eval()
            with torch.no_grad():
                val_logits = head(encoder(node_features, edge_index))
                val_loss = criterion(val_logits[val_mask], labels[val_mask])
                val_prob = torch.sigmoid(val_logits[val_mask]).cpu().numpy().ravel()
                final_val_auc = float(roc_auc_score(labels[val_mask].numpy().ravel(), val_prob))

            mlflow.log_metric("train_loss", float(train_loss.item()), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss.item()), step=epoch)
            mlflow.log_metric("val_auc", final_val_auc, step=epoch)
            print(
                f"[INFO] epoch={epoch} train_loss={train_loss.item():.4f} "
                f"val_loss={val_loss.item():.4f} val_auc={final_val_auc:.4f}"
            )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    print(f"[INFO] Early stopping at epoch {epoch}.")
                    break

    encoder.load_state_dict(best_state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), PROCESSED_DIR / "gnn_encoder.pt")
    print(f"[INFO] Final val AUC: {final_val_auc:.4f}")
    print(f"[INFO] Final graph stats: {stats}")
    return final_val_auc, stats


def _encode_text_split(split_name: str, encoder: Any, text_df: pd.DataFrame) -> Any:
    """Encode one processed split and save its text embeddings."""
    import torch

    from src.data.text_preprocessor import align_texts_with_tabular

    split_path = PROCESSED_DIR / f"{split_name}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing processed split: {split_path}")

    split_df = pd.read_parquet(split_path)
    texts = align_texts_with_tabular(text_df, split_df)
    embeddings = []

    for start in range(0, len(texts), 64):
        chunk = texts[start : start + 64]
        with torch.no_grad():
            embeddings.append(encoder.encode_texts(chunk).detach().cpu())

    output = torch.cat(embeddings, dim=0)
    output_path = PROCESSED_DIR / f"text_embeddings_{split_name}.pt"
    torch.save(output, output_path)
    return output, output_path


def run_text_stage() -> dict[str, tuple[int, ...]]:
    """Run the Day 4 frozen text embedding generation stage.

    Returns:
        Mapping of split names to embedding tensor shapes.
    """
    import mlflow

    from src.data.text_preprocessor import load_loan_descriptions
    from src.models.text_encoder import TextEncoder

    if not SYNTHETIC_TEXT_PATH.exists():
        raise FileNotFoundError(f"Missing synthetic text CSV: {SYNTHETIC_TEXT_PATH}")

    start_time = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    text_df = load_loan_descriptions(str(SYNTHETIC_TEXT_PATH))
    encoder = TextEncoder(freeze=True)

    with mlflow.start_run():
        train_embeddings, train_path = _encode_text_split("train", encoder, text_df)
        val_embeddings, val_path = _encode_text_split("val", encoder, text_df)
        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(val_path))

    elapsed = time.perf_counter() - start_time
    total = train_embeddings.shape[0] + val_embeddings.shape[0]
    shapes = {
        "train": tuple(train_embeddings.shape),
        "val": tuple(val_embeddings.shape),
    }
    print(
        f"[INFO] Text embeddings generated: total={total} "
        f"train_shape={shapes['train']} val_shape={shapes['val']} "
        f"time_taken={elapsed:.2f}s"
    )
    return shapes


def _load_pickle(path: Path) -> Any:
    """Load a pickle artifact from disk."""
    with path.open("rb") as file:
        return pickle.load(file)


def _make_lstm_embeddings(df: pd.DataFrame, state_path: Path) -> Any:
    """Create LSTM embeddings for one split using a saved encoder state."""
    import torch

    from src.models.lstm_encoder import LSTMEncoder

    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    encoder.load_state_dict(torch.load(state_path, map_location="cpu"))
    encoder.eval()
    with torch.no_grad():
        sequences = torch.FloatTensor(create_synthetic_time_series(df))
        return encoder(sequences).detach().cpu()


def _make_gnn_embeddings(df: pd.DataFrame, state_path: Path) -> Any:
    """Create GraphSAGE embeddings for one split using a saved encoder state."""
    import torch

    from src.data.graph_builder import build_borrower_graph
    from src.models.gnn_encoder import GraphSAGEEncoder

    edge_index, node_features = build_borrower_graph(df)
    encoder = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    encoder.load_state_dict(torch.load(state_path, map_location="cpu"))
    encoder.eval()
    with torch.no_grad():
        return encoder(node_features, edge_index).detach().cpu()


def _fusion_metrics(labels: Any, probabilities: Any) -> dict[str, float]:
    """Compute fusion validation metrics."""
    y_true = labels.detach().cpu().numpy().ravel()
    y_prob = probabilities.detach().cpu().numpy().ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_fusion_stage() -> dict[str, float]:
    """Run the Day 5 late-fusion classifier training stage.

    Returns:
        Validation metrics dictionary.
    """
    import mlflow
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.fusion import LateFusionClassifier, get_xgb_leaf_embeddings

    required_paths = {
        "train": PROCESSED_DIR / "train.parquet",
        "val": PROCESSED_DIR / "val.parquet",
        "xgb": PROCESSED_DIR / "xgb_baseline.pkl",
        "lstm": PROCESSED_DIR / "lstm_encoder.pt",
        "gnn": PROCESSED_DIR / "gnn_encoder.pt",
        "text_train": PROCESSED_DIR / "text_embeddings_train.pt",
        "text_val": PROCESSED_DIR / "text_embeddings_val.pt",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fusion artifacts: {missing}")

    torch.manual_seed(42)
    train_df = pd.read_parquet(required_paths["train"])
    val_df = pd.read_parquet(required_paths["val"])
    xgb_model = _load_pickle(required_paths["xgb"])

    X_train, y_train_series = _feature_target(train_df)
    X_val, y_val_series = _feature_target(val_df)
    train_labels = torch.FloatTensor(y_train_series.to_numpy()).view(-1, 1)
    val_labels = torch.FloatTensor(y_val_series.to_numpy()).view(-1, 1)

    train_tabular = get_xgb_leaf_embeddings(xgb_model, X_train)
    val_tabular = get_xgb_leaf_embeddings(xgb_model, X_val)
    train_lstm = _make_lstm_embeddings(train_df, required_paths["lstm"])
    val_lstm = _make_lstm_embeddings(val_df, required_paths["lstm"])
    train_gnn = _make_gnn_embeddings(train_df, required_paths["gnn"])
    val_gnn = _make_gnn_embeddings(val_df, required_paths["gnn"])
    train_text = torch.load(required_paths["text_train"], map_location="cpu").float()
    val_text = torch.load(required_paths["text_val"], map_location="cpu").float()

    train_dataset = TensorDataset(
        train_tabular,
        train_lstm,
        train_gnn,
        train_text,
        train_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = LateFusionClassifier()
    pos_count = train_labels.sum()
    neg_count = train_labels.numel() - pos_count
    pos_weight = neg_count / pos_count.clamp(min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = model.state_dict()
    epochs_without_improvement = 0
    final_metrics: dict[str, float] = {}

    with mlflow.start_run():
        mlflow.log_params(
            {
                "stage": "fusion",
                "batch_size": 256,
                "epochs": 30,
                "patience": 7,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "pos_weight": float(pos_weight.item()),
            }
        )

        for epoch in range(1, 31):
            model.train()
            train_loss_total = 0.0
            train_examples = 0
            for tabular, lstm, gnn, text, labels in train_loader:
                optimizer.zero_grad()
                logits = model(tabular, lstm, gnn, text)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                train_loss_total += loss.item() * batch_size
                train_examples += batch_size

            train_loss = train_loss_total / train_examples

            model.eval()
            with torch.no_grad():
                val_logits = model(val_tabular, val_lstm, val_gnn, val_text)
                val_loss = criterion(val_logits, val_labels)
                val_prob = torch.sigmoid(val_logits)
                final_metrics = _fusion_metrics(val_labels, val_prob)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", float(val_loss.item()), step=epoch)
            mlflow.log_metric("val_auc", final_metrics["roc_auc"], step=epoch)
            mlflow.log_metric("val_pr_auc", final_metrics["pr_auc"], step=epoch)
            print(
                f"[INFO] epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_loss.item():.4f} val_auc={final_metrics['roc_auc']:.4f} "
                f"val_pr_auc={final_metrics['pr_auc']:.4f}"
            )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 7:
                    print(f"[INFO] Early stopping at epoch {epoch}.")
                    break

    model.load_state_dict(best_state)
    output_path = PROCESSED_DIR / "fusion_model.pt"
    torch.save(model.state_dict(), output_path)
    print(
        f"[INFO] Fusion val AUC: {final_metrics['roc_auc']:.4f} "
        f"PR-AUC: {final_metrics['pr_auc']:.4f} "
        f"F1@0.5: {final_metrics['f1@0.5']:.4f}"
    )
    return final_metrics


def main() -> None:
    """Parse CLI args and run the requested training stage."""
    parser = argparse.ArgumentParser(description="Credit risk training loop.")
    parser.add_argument("--stage", choices=["xgb", "lstm", "gnn", "text", "fusion"], required=True)
    args = parser.parse_args()

    if args.stage == "xgb":
        run_xgb_stage()
    elif args.stage == "lstm":
        run_lstm_stage()
    elif args.stage == "gnn":
        run_gnn_stage()
    elif args.stage == "text":
        run_text_stage()
    elif args.stage == "fusion":
        run_fusion_stage()


if __name__ == "__main__":
    main()
