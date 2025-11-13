"""Train and test an LSTM model for EMG gesture recognition on Dataseet.

Inputs (in this folder):
- label.txt                 lines like: '1 - one'
- <label>_processed.csv     each file contains continuous EMG samples with 4 columns

This script:
- Builds fixed-length windows over each CSV (sequence_length, channels)
- Trains a PyTorch LSTM classifier
- Evaluates on a held-out test split
- Saves: lstm_model.pt, identity.json, and prints metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def parse_label_file(label_file: Path) -> List[Tuple[int, str]]:
    mapping: List[Tuple[int, str]] = []
    with label_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "-" not in line:
                raise ValueError(f"Invalid label line: {line}")
            idx_str, name = line.split("-", maxsplit=1)
            mapping.append((int(idx_str.strip()), name.strip()))
    mapping.sort(key=lambda x: x[0])
    return mapping


def build_windows(
    array_2d: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Convert [N, C] to [num_windows, window_size, C] with given stride."""
    num_samples, num_channels = array_2d.shape
    if num_samples < window_size:
        return np.empty((0, window_size, num_channels), dtype=array_2d.dtype)
    windows = []
    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(array_2d[start:end])
    if not windows:
        return np.empty((0, window_size, num_channels), dtype=array_2d.dtype)
    return np.stack(windows, axis=0)


class EMGSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        label_mapping: Sequence[Tuple[int, str]],
        window_size: int,
        stride: int,
        standardize: bool = True,
        fit_stats: bool = True,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.stride = stride
        self.standardize = standardize

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        identity: Dict[str, str] = {}

        for class_idx, (_id_from_file, label_name) in enumerate(label_mapping):
            csv_path = dataset_dir / f"{label_name}_processed.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing processed file for label '{label_name}': {csv_path}")
            df = pd.read_csv(csv_path)
            arr = df.to_numpy(dtype=np.float32)  # [N, C]

            # windowing
            Xw = build_windows(arr, window_size, stride)  # [M, T, C]
            if Xw.shape[0] == 0:
                continue
            yw = np.full((Xw.shape[0],), class_idx, dtype=np.int64)

            X_list.append(Xw)
            y_list.append(yw)
            identity[str(class_idx)] = label_name

        if not X_list:
            raise RuntimeError("No windows built; check window_size/stride and CSV contents.")

        X = np.concatenate(X_list, axis=0)  # [Total, T, C]
        y = np.concatenate(y_list, axis=0)  # [Total]

        # Standardize per-channel over entire sequence set
        if standardize:
            if fit_stats:
                # compute mean/std over all time and windows per channel
                # reshape to [Total*T, C] for stats
                flat = X.reshape(-1, X.shape[-1])
                self.mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
                self.std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
                self.std[self.std == 0] = 1.0
            else:
                if mean is None or std is None:
                    raise ValueError("Must provide mean and std when fit_stats=False.")
                self.mean = mean.astype(np.float32)
                self.std = std.astype(np.float32)
            X = (X - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        else:
            self.mean = np.zeros((X.shape[-1],), dtype=np.float32)
            self.std = np.ones((X.shape[-1],), dtype=np.float32)

        self.X = X  # float32
        self.y = y  # int64
        self.identity = identity

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # [T, C]
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        out, _ = self.lstm(x)  # [B, T, H*(2?)]
        last = out[:, -1, :]   # [B, H*(2?)]
        logits = self.fc(last) # [B, K]
        return logits


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.argmax(dim=1) == target).float().mean().item()


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def stratified_equal_split_indices(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """Return train/val indices with equal samples per class.

    - Find counts per class
    - Use the minimum class count as baseline
    - Split per class into equal train/val based on val_ratio
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class_indices = {c: np.where(labels == c)[0] for c in classes}
    per_class_counts = {c: len(per_class_indices[c]) for c in classes}
    min_count = int(min(per_class_counts.values()))
    if min_count < 2:
        raise RuntimeError("Not enough samples in at least one class for a split.")

    n_val = max(1, int(round(min_count * val_ratio)))
    n_train = min_count - n_val
    if n_train < 1:
        n_train = 1
        n_val = min_count - n_train

    train_idx_list: List[int] = []
    val_idx_list: List[int] = []
    picked_train_counts: Dict[int, int] = {}
    picked_val_counts: Dict[int, int] = {}

    for c in classes:
        idxs = per_class_indices[c]
        rng.shuffle(idxs)
        choose = idxs[:min_count]
        val_part = choose[:n_val]
        train_part = choose[n_val:n_val + n_train]
        train_idx_list.extend(train_part.tolist())
        val_idx_list.extend(val_part.tolist())
        picked_train_counts[int(c)] = len(train_part)
        picked_val_counts[int(c)] = len(val_part)

    rng.shuffle(train_idx_list)
    rng.shuffle(val_idx_list)
    return np.array(train_idx_list), np.array(val_idx_list), picked_train_counts, picked_val_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Test LSTM on EMG Dataseet.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent, help="Directory with processed CSVs and label.txt")
    parser.add_argument("--label-file", type=Path, default=Path(__file__).parent / "label.txt", help="Path to label.txt")
    parser.add_argument("--sequence-length", type=int, default=64, help="Window size (timesteps)")
    parser.add_argument("--stride", type=int, default=16, help="Window stride")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers (if num_layers>1)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label_mapping = parse_label_file(args.label_file)
    num_classes = len(label_mapping)

    # Build full dataset and compute normalization on full set,
    # then split into train/test and reapply stats to both sets consistently.
    full_ds = EMGSequenceDataset(
        dataset_dir=args.dataset_dir,
        label_mapping=label_mapping,
        window_size=args.sequence_length,
        stride=args.stride,
        standardize=True,
        fit_stats=True,
    )

    # Report original per-class window counts
    unique, counts = np.unique(full_ds.y, return_counts=True)
    print("Per-class window counts (before balancing):")
    for c, cnt in zip(unique, counts):
        print(f"  class {int(c)}: {int(cnt)}")

    # Stratified equal split based on per-window labels
    train_idx, val_idx, picked_train, picked_val = stratified_equal_split_indices(
        labels=full_ds.y, val_ratio=args.val_split, seed=args.seed
    )
    print("Picked per-class counts (train):", picked_train)
    print("Picked per-class counts (val):", picked_val)

    # Subset datasets
    from torch.utils.data import Subset  # local import to avoid clutter at top
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    input_size = full_ds.X.shape[-1]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)  # [B, T, C]
            yb = yb.to(device)  # [B]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), yb)
            train_batches += 1
        train_loss /= max(train_batches, 1)
        train_acc /= max(train_batches, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb)
                val_batches += 1
        val_loss /= max(val_batches, 1)
        val_acc /= max(val_batches, 1)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    # Final evaluation metrics and confusion matrices
    def evaluate(loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        preds: List[int] = []
        trues: List[int] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        acc = (y_pred == y_true).mean().item()
        cm = confusion_matrix_np(y_true, y_pred, num_classes)
        return acc, cm, y_pred

    train_acc, train_cm, _ = evaluate(train_loader)
    val_acc, val_cm, _ = evaluate(val_loader)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Train confusion matrix:\n{train_cm}")
    # Row-normalized matrices (per-class recall)
    train_row_sums = train_cm.sum(axis=1, keepdims=True).clip(min=1)
    train_cm_norm = train_cm / train_row_sums
    print(f"Train confusion matrix (row-normalized):\n{train_cm_norm}")
    print(f"Test accuracy: {val_acc:.4f}")
    print(f"Test confusion matrix:\n{val_cm}")
    val_row_sums = val_cm.sum(axis=1, keepdims=True).clip(min=1)
    val_cm_norm = val_cm / val_row_sums
    print(f"Test confusion matrix (row-normalized):\n{val_cm_norm}")

    # Save artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "bidirectional": args.bidirectional,
            "dropout": args.dropout,
            "num_classes": num_classes,
            "sequence_length": args.sequence_length,
            "mean": full_ds.mean,
            "std": full_ds.std,
        },
        Path(__file__).parent / "lstm_model.pt",
    )
    with (Path(__file__).parent / "identity.json").open("w", encoding="utf-8") as f:
        json.dump(full_ds.identity, f, ensure_ascii=False, indent=2)
    print("Saved model to:", Path(__file__).parent / "lstm_model.pt")
    print("Saved identity to:", Path(__file__).parent / "identity.json")


if __name__ == "__main__":
    main()

