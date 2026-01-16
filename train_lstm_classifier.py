"""
Train an LSTM (next-token LM) on integer sequences with dynamic vocab shift.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ARTIFACT_ROOT = Path("artifacts")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["train"], data["val"], data["test"]


def shift_and_vocab(
    arrays: Iterable[np.ndarray]
) -> tuple[list[np.ndarray], int, int]:
    """
    Shift all arrays so min is 0; return shifted arrays, vocab size, and shift.
    """
    arrays = list(arrays)
    global_min = min(a.min() for a in arrays)
    global_max = max(a.max() for a in arrays)
    shifted = [(a - global_min).astype(np.int64) for a in arrays]
    vocab_size = int(global_max - global_min + 1)
    return shifted, vocab_size, int(global_min)


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seqs: torch.Tensor) -> None:
        # next-token: input is seq[:-1], target is seq[1:]
        self.inputs = seqs[:, :-1]
        self.targets = seqs[:, 1:]

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class LstmModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        out, _ = self.lstm(emb)
        return self.linear(out)


@dataclass
class Metrics:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
            total_loss += loss.item() * targets.numel()

    return Metrics(
        loss=total_loss / total_tokens,
        accuracy=correct / total_tokens
    )


def create_loader(
    tensor: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        SequenceDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train LSTM LM with dynamic vocab shift."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Split NPZ with train/val/test"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    train_np, val_np, test_np = load_split(args.data)
    shifted_arrays, vocab_size, shift = \
        shift_and_vocab([train_np, val_np, test_np])
    train_np, val_np, test_np = shifted_arrays

    # log vocab info
    print(
        f"Shifted by {shift}; "
        f"vocab_size={vocab_size}; "
        f"min={min(a.min() for a in shifted_arrays)}, "
        f"max={max(a.max() for a in shifted_arrays)}"
    )

    # tensors
    train_t = torch.from_numpy(train_np)
    val_t = torch.from_numpy(val_np)
    test_t = torch.from_numpy(test_np)

    device = get_device()
    model = LstmModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = create_loader(
        train_t,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = create_loader(
        val_t,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = create_loader(
        test_t,
        batch_size=args.batch_size,
        shuffle=False
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = \
            run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = \
            run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.3f}"
        )
        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion, None, device)
    print(
        "Test: "
        f"loss={test_metrics.loss:.4f} "
        f"acc={test_metrics.accuracy:.3f}"
    )

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_ROOT / f"{args.data.stem}-lstm.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
