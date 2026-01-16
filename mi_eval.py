"""
Membership inference evaluation for trained LSTM checkpoints.

Given two sets of integer sequences (members vs non-members), compute a
per-sequence average log-probability under the LSTM, derive distributions,
the ROC/AUC curve, and the TPR achieved at a target FPR, and save the plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from train_lstm_classifier import LstmModel, get_device


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seqs: torch.Tensor) -> None:
        self.inputs = seqs[:, :-1]
        self.targets = seqs[:, 1:]

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_loader(arr: np.ndarray, batch_size: int) -> DataLoader:
    t = torch.from_numpy(arr.astype(np.int64))
    return DataLoader(SequenceDataset(t), batch_size=batch_size, shuffle=False)


def load_sequences(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["sequences"].astype(np.int64)


def scores_from_loader(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Average log-probability per sequence."""
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            seq_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            seq_mean = seq_logp.mean(dim=1)
            scores.extend(seq_mean.cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def load_model(
    model_path: Path,
    state: dict[str, torch.Tensor],
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> LstmModel:
    model = LstmModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    ckpt_vocab = state["embedding.weight"].shape[0]
    if ckpt_vocab != vocab_size:
        model = LstmModel(
            vocab_size=ckpt_vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def shift_sequences(arr: np.ndarray, shift: int, vocab: int) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.int64)
    shifted = arr - shift
    shifted = np.clip(shifted, 0, vocab - 1)
    return shifted.astype(np.int64)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def evaluate_membership(
    model_path: Path,
    members_path: Path,
    nonmembers_path: Path,
    batch_cap: int = 0,
    batch_size: int = 128,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
    target_fpr: float = 0.01,
    shift: int | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray | float | int]:
    """Load model + data, compute MI scores, and return metrics for plotting."""
    device = get_device()

    members = load_sequences(members_path)
    nonmembers = load_sequences(nonmembers_path)
    if batch_cap > 0:
        rng = np.random.default_rng(seed)
        mem_idx = rng.choice(len(members), size=min(batch_cap, len(members)), replace=False)
        non_idx = rng.choice(len(nonmembers), size=min(batch_cap, len(nonmembers)), replace=False)
        members = members[mem_idx]
        nonmembers = nonmembers[non_idx]

    overall_min = int(min(members.min(), nonmembers.min())) if shift is None else shift
    state = torch.load(model_path, map_location="cpu")
    vocab = int(state["embedding.weight"].shape[0])

    model = load_model(model_path, state, vocab, embedding_dim, hidden_dim, num_layers, dropout, device)

    members_shifted = shift_sequences(members, overall_min, vocab)
    nonmembers_shifted = shift_sequences(nonmembers, overall_min, vocab)

    loader_mem = create_loader(members_shifted, batch_size)
    loader_non = create_loader(nonmembers_shifted, batch_size)

    scores_mem = scores_from_loader(model, loader_mem, device)
    scores_non = scores_from_loader(model, loader_non, device)
    y_true = np.array([1] * len(scores_mem) + [0] * len(scores_non), dtype=np.int32)
    scores = np.concatenate([scores_mem, scores_non], axis=0)
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "y_true": y_true,
        "scores": scores,
        "scores_mem": scores_mem,
        "scores_non": scores_non,
        "auc": float(auc),
        "tpr_at_fpr": float(tpr_at_fpr(y_true, scores, target_fpr)),
        "target_fpr": float(target_fpr),
        "n_mem": len(members),
        "n_non": len(nonmembers),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_score_hist(scores_mem: np.ndarray, scores_non: np.ndarray, label: str) -> None:
    """Overlay histograms for member vs non-member scores."""
    plt.figure(figsize=(6, 4))
    plt.hist(scores_mem, bins=40, alpha=0.6, label="members")
    plt.hist(scores_non, bins=40, alpha=0.6, label="non-members")
    plt.xlabel("Score (negative loss)")
    plt.ylabel("Count")
    plt.title(f"Score Distributions for MI on {label}")
    plt.legend()
    plt.tight_layout()


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, label: str) -> None:
    """Plot ROC curve for membership inference scores."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="MI ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC for MI on {label}")
    plt.legend()
    plt.tight_layout()


def plot_tpr_point(target_fpr: float, achieved_tpr: float, label: str) -> None:
    """Plot a single point highlighting TPR@target FPR."""
    plt.figure(figsize=(4, 4))
    plt.scatter([target_fpr], [achieved_tpr], color="tab:red", s=80, label=f"TPR@FPR={target_fpr:.3f}")
    plt.axvline(target_fpr, color="tab:red", linestyle="--", linewidth=1)
    plt.axhline(achieved_tpr, color="tab:red", linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"TPR at FPR={target_fpr:.3f} for MI on {label}")
    plt.legend()
    plt.tight_layout()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Membership inference evaluation for LSTM checkpoints.")
    parser.add_argument("--model", type=Path, required=True, help=".pt checkpoint trained via train_lstm_classifier.py")
    parser.add_argument("--members", type=Path, required=True, help="NPZ with member sequences (int arrays)")
    parser.add_argument("--nonmembers", type=Path, required=True, help="NPZ with non-member sequences (int arrays)")
    parser.add_argument("--batch", type=int, default=0, help="Optional cap on sequences sampled from each set (0 = all)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for scoring dataloaders")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shift", type=int, help="Optional token shift override; defaults to min token across both sets")
    parser.add_argument("--fpr", type=float, default=0.01, help="Target FPR for TPR@FPR plot")
    parser.add_argument("--out-dir", type=Path, default=Path("mi-plots"), help="Directory to save ROC/hist/TPR plots")
    parser.add_argument("--label", type=str, help="Human-readable description of the model/data (used in plot titles)")
    args = parser.parse_args(argv)

    summary = evaluate_membership(
        model_path=args.model,
        members_path=args.members,
        nonmembers_path=args.nonmembers,
        batch_cap=args.batch,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        target_fpr=args.fpr,
        shift=args.shift,
    )
    print(f"AUC: {summary['auc']:.4f}")
    print(f"TPR@FPR={summary['target_fpr']:.3f}: {summary['tpr_at_fpr']:.4f}")
    print(f"Members: n={summary['n_mem']}, mean={summary['scores_mem'].mean():.4f}, std={summary['scores_mem'].std():.4f}")
    print(f"Non-members: n={summary['n_non']}, mean={summary['scores_non'].mean():.4f}, std={summary['scores_non'].std():.4f}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or f"{args.model.stem}"

    plot_score_hist(summary["scores_mem"], summary["scores_non"], label)
    hist_path = out_dir / "scores_hist.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    plot_roc_curve(summary["y_true"], summary["scores"], label)
    roc_path = out_dir / "roc.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    plot_tpr_point(summary["target_fpr"], summary["tpr_at_fpr"], label)
    tpr_path = out_dir / "tpr_at_fpr.png"
    plt.savefig(tpr_path, dpi=200)
    plt.close()

    print(f"Saved histogram to {hist_path}")
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved TPR@FPR plot to {tpr_path}")


if __name__ == "__main__":
    main()
