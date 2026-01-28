"""Simple LSTM training script for 1..100 integer sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lstm_data import (
    load_sequences,
    map_to_class_indices,
    resolve_npz_files,
    train_val_test_split,
)

import argparse

ARTIFACT_ROOT = Path("artifacts")
NUM_CLASSES = 100


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seqs: torch.Tensor) -> None:
        self.inputs = seqs[:, :-1]
        self.targets = seqs[:, 1:]

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class LstmModel(nn.Module):
    # lstm settings for 16 sequences of length 16
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        outputs, _ = self.lstm(embedded)
        logits = self.linear(outputs)
        return logits


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
    correct_tokens = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct_tokens += (predictions == targets).sum().item()
            token_count = targets.numel()
            total_tokens += token_count
            total_loss += loss.item() * token_count

    average_loss = total_loss / total_tokens
    accuracy = correct_tokens / total_tokens
    return Metrics(loss=average_loss, accuracy=accuracy)


def create_dataloader(
    tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    dataset = SequenceDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_data(
    paths: Iterable[Path]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = load_sequences(paths)
    mapped_seqs = map_to_class_indices(seqs)
    dataset = train_val_test_split(mapped_seqs)
    to_tensor = lambda arr: torch.from_numpy(arr.astype("int64"))
    return (
        to_tensor(dataset.train),
        to_tensor(dataset.val),
        to_tensor(dataset.test)
    )


def infer_sequence_metadata(npz_files: Sequence[Path]) -> tuple[str, int]:
    """Derive prefix and sequence length from artifact filenames."""
    if not npz_files:
        raise ValueError("expected at least one NPZ file")

    prefix_and_len = {tuple(path.stem.split("-")[:2]) for path in npz_files}
    if len(prefix_and_len) != 1:
        details = ", ".join("-".join(parts) for parts in prefix_and_len)
        raise ValueError(
            "data files must share a common prefix and sequence length; "
            f"found: {details}"
        )

    prefix, seq_len_str = prefix_and_len.pop()
    try:
        seq_len = int(seq_len_str)
    except ValueError as exc:
        raise ValueError(
            f"unexpected sequence length component {seq_len_str!r} in {npz_files[0].name}"
        ) from exc

    return prefix, seq_len

def main() -> int:
    parser = argparse.ArgumentParser(description="Train an LSTM on integer sequences")
    parser.add_argument(
        "--data-filepath", "-p",
        type=Path,
        required=True,
        help="NPZ artifact containing training sequences",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    args = parser.parse_args()

    npz_files = resolve_npz_files([args.data_filepath])
    prefix, seq_len = infer_sequence_metadata(npz_files)

    train, val, test = prepare_data(npz_files)
    device = get_device()

    model = LstmModel(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = create_dataloader(train, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val, batch_size=4, shuffle=False)
    test_loader = create_dataloader(test, batch_size=4, shuffle=False)

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        train_metrics = \
            run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)

        message = (
            f"Epoch {epoch:02d}: "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.3f}"
        )
        print(message)

        test_metrics = run_epoch(model, test_loader, criterion, None, device)
        print(
            f"Test: "
            f"loss={test_metrics.loss:.4f} "
            f"acc={test_metrics.accuracy:.3f}"
        )

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_ROOT / f"{prefix}-{seq_len}-lstm.pt"

    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
