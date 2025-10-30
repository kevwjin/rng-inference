"""Simple LSTM training script for 1..100 integer sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lstm_data import (
    load_sequences,
    map_to_class_indices,
    train_val_test_split,
)

SAMPLES_ROOT = Path("samples")
DEFAULT_DATA_DIRS: tuple[Path, ...] = (SAMPLES_ROOT / "en-16",)
NUM_CLASSES = 100


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seqs: torch.Tensor) -> None:
        if seqs.ndim != 2:
            raise ValueError("sequences tensor must be 2D")
        if seqs.shape[1] < 2:
            raise ValueError(
                "sequences must have length >= 2 for next-token training")
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
    dirs: Iterable[Path]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = load_sequences(dirs)
    mapped_seqs = map_to_class_indices(seqs)
    dataset = train_val_test_split(mapped_seqs)
    to_tensor = lambda arr: torch.from_numpy(arr.astype("int64"))
    return (
        to_tensor(dataset.train),
        to_tensor(dataset.val),
        to_tensor(dataset.test)
    )


def main() -> int:
    dirs = [path for path in DEFAULT_DATA_DIRS if path.exists()]
    if not dirs:
        raise FileNotFoundError("no data directories found under samples/.")

    train, val, test = prepare_data(dirs)
    device = get_device()

    model = LstmModel(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = create_dataloader(train, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val, batch_size=4, shuffle=False)
    test_loader = create_dataloader(test, batch_size=4, shuffle=False)

    epochs = 10
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

    torch.save(model.state_dict(), "lstm_model.pt")
    print("Saved model to lstm_model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
