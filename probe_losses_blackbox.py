"""Black-box probe: given probes already in model token space, compute average loss."""
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


def create_loader(arr: np.ndarray, batch_size: int) -> DataLoader:
    t = torch.from_numpy(arr.astype(np.int64))
    return DataLoader(SequenceDataset(t), batch_size=batch_size, shuffle=False)


def load_sequences(path: Path, count: int | None, rng: np.random.Generator) -> np.ndarray:
    with np.load(path) as data:
        seqs = data["sequences"].astype(np.int64)
    if count is not None and count < seqs.shape[0]:
        idx = rng.choice(seqs.shape[0], size=count, replace=False)
        seqs = seqs[idx]
    return seqs


def load_model(model_path: Path, device: torch.device) -> tuple[LstmModel, int]:
    state = torch.load(model_path, map_location=device)
    vocab_size = state["embedding.weight"].shape[0]
    model = LstmModel(vocab_size=vocab_size, embedding_dim=state["embedding.weight"].shape[1],
                      hidden_dim=state["lstm.weight_hh_l0"].shape[1], num_layers=1, dropout=0.0)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, vocab_size


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
            total_loss += loss.item() * targets.numel()
    return total_loss / total_tokens, correct / total_tokens


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Black-box probe: assume probes already in model token space.")
    parser.add_argument("--en-data", type=Path, required=True, help="NPZ with 'sequences' already tokenized for the model")
    parser.add_argument("--zh-data", type=Path, required=True)
    parser.add_argument("--prng-data", type=Path, required=True)
    parser.add_argument("--en-model", type=Path, required=True)
    parser.add_argument("--zh-model", type=Path, required=True)
    parser.add_argument("--prng-model", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, help="Optional cap per probe set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-json", type=Path, help="Optional JSON output")
    args = parser.parse_args(argv)

    device = get_device()
    rng = np.random.default_rng(42)
    criterion = nn.CrossEntropyLoss()

    probes = {
        "en": load_sequences(args.en_data, args.probe_count, rng),
        "zh": load_sequences(args.zh_data, args.probe_count, rng),
        "prng": load_sequences(args.prng_data, args.probe_count, rng),
    }

    models = {
        "en": load_model(args.en_model, device),
        "zh": load_model(args.zh_model, device),
        "prng": load_model(args.prng_model, device),
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, (model, vocab) in models.items():
        results[model_name] = {}
        for probe_name, arr in probes.items():
            # shift probes to start at zero and clip to vocab range
            arr_shift = arr - arr.min()
            arr_shift = np.clip(arr_shift, 0, vocab - 1)
            loader = create_loader(arr_shift, args.batch_size)
            loss, acc = run_epoch(model, loader, criterion, device)
            print(f"{model_name} on {probe_name}: loss={loss:.4f} acc={acc:.3f}")
            results[model_name][probe_name] = {"loss": loss, "acc": acc}

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
