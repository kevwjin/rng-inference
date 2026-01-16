"""Simulate federated clients per source (EN/ZH/PRNG) and probe loss attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
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


def create_loader(arr: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    t = torch.from_numpy(arr.astype(np.int64))
    return DataLoader(SequenceDataset(t), batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_tokens, correct = 0.0, 0, 0
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
    return total_loss / total_tokens, correct / total_tokens


def shift_and_vocab(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    gmin = min(a.min() for a in arrays)
    gmax = max(a.max() for a in arrays)
    shifted = [(a - gmin).astype(np.int64) for a in arrays]
    vocab = int(gmax - gmin + 1)
    return shifted, vocab, int(gmin)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate federated clients and probe attribution.")
    parser.add_argument("--en-data", type=Path, default=Path("artifacts/splits/en-split.npz"))
    parser.add_argument("--zh-data", type=Path, default=Path("artifacts/splits/zh-split.npz"))
    parser.add_argument("--prng-data", type=Path, default=Path("artifacts/splits/prng-split.npz"))
    parser.add_argument("--clients-per-source", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-count", type=int, default=500)
    parser.add_argument("--out-json", type=Path, help="Optional JSON path to save per-client probe losses")
    args = parser.parse_args(argv)

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)

    def load_split(path: Path):
        with np.load(path) as data:
            return data["train"], data["val"], data["test"]

    # Load and shift all datasets with a shared vocab
    en_tr, en_val, en_te = load_split(args.en_data)
    zh_tr, zh_val, zh_te = load_split(args.zh_data)
    prng_tr, prng_val, prng_te = load_split(args.prng_data)
    shifted_arrays, vocab_size, shift = shift_and_vocab([en_tr, zh_tr, prng_tr, en_val, zh_val, prng_val, en_te, zh_te, prng_te])
    en_tr, zh_tr, prng_tr, en_val, zh_val, prng_val, en_te, zh_te, prng_te = shifted_arrays
    print(f"Shared shift={shift}, vocab={vocab_size}")

    # Build probes (sample from test)
    def sample_probe(arr: np.ndarray) -> np.ndarray:
        if args.probe_count and args.probe_count < arr.shape[0]:
            idx = rng.choice(arr.shape[0], size=args.probe_count, replace=False)
            return arr[idx]
        return arr

    probes = {
        "en": sample_probe(en_te),
        "zh": sample_probe(zh_te),
        "prng": sample_probe(prng_te),
    }

    # Function to train a client model on local data
    def train_client(data: np.ndarray) -> LstmModel:
        model = LstmModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = create_loader(data, args.batch_size, shuffle=True)
        for _ in range(args.local_epochs):
            run_epoch(model, loader, criterion, opt, device)
        return model

    # Train clients
    clients = []
    for source, data in [("en", en_tr), ("zh", zh_tr), ("prng", prng_tr)]:
        for i in range(args.clients_per_source):
            model = train_client(data)
            clients.append((source, model))
    print(f"Trained {len(clients)} clients (per-source={args.clients_per_source})")

    results = []
    # Evaluate probe losses per client
    for idx, (source, model) in enumerate(clients, 1):
        print(f"Client {idx} (source={source})")
        for probe_name, arr in probes.items():
            loader = create_loader(arr, args.batch_size, shuffle=False)
            loss, acc = run_epoch(model, loader, criterion, optimizer=None, device=device)
            print(f"  Probe {probe_name}: loss={loss:.4f} acc={acc:.3f}")
            results.append({"client": f"{source}_{idx}", "source": source, "probe": probe_name, "loss": loss, "acc": acc})
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved client probe losses to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
