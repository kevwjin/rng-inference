"""Compute cross-probe losses for trained LSTMs on EN/ZH/PRNG splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from train_lstm_classifier import LstmModel, load_split, get_device, SequenceDataset
from torch.utils.data import DataLoader


def create_loader(arr: np.ndarray, batch_size: int) -> DataLoader:
    t = torch.from_numpy(arr.astype(np.int64))
    return DataLoader(SequenceDataset(t), batch_size=batch_size, shuffle=False)


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


def load_model(model_path: Path, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float, device: torch.device) -> LstmModel:
    model = LstmModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    state = torch.load(model_path, map_location=device)
    # If vocab sizes mismatch, rebuild model with checkpoint vocab
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


def shift_with_global_min(arr: np.ndarray, global_min: int) -> np.ndarray:
    return (arr - global_min).astype(np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-probe losses across EN/ZH/PRNG models.")
    parser.add_argument("--en-split", type=Path, default=Path("artifacts/splits/en-split.npz"))
    parser.add_argument("--zh-split", type=Path, default=Path("artifacts/splits/zh-split.npz"))
    parser.add_argument("--prng-split", type=Path, default=Path("artifacts/splits/prng-split.npz"))
    parser.add_argument("--en-model", type=Path, default=Path("artifacts/en-split-lstm.pt"))
    parser.add_argument("--zh-model", type=Path, default=Path("artifacts/zh-split-lstm.pt"))
    parser.add_argument("--prng-model", type=Path, default=Path("artifacts/prng-split-lstm.pt"))
    parser.add_argument("--probe-count", type=int, default=500, help="Sequences per probe set (sampled from test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out-json", type=Path, help="Optional JSON path to save results")
    args = parser.parse_args(argv)

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)

    def dataset_min_max(split_path: Path):
        tr, val, te = load_split(split_path)
        gmin = int(min(tr.min(), val.min(), te.min()))
        gmax = int(max(tr.max(), val.max(), te.max()))
        return gmin, gmax

    # compute training shift/vocab for each dataset
    specs = {}
    for name, split in [("en", args.en_split), ("zh", args.zh_split), ("prng", args.prng_split)]:
        gmin, gmax = dataset_min_max(split)
        specs[name] = {"shift": gmin, "vocab": gmax - gmin + 1}
        print(f"{name} training range=({gmin},{gmax}), vocab={specs[name]['vocab']}")

    def sample_probe(split_path: Path):
        _, _, te = load_split(split_path)
        if args.probe_count and args.probe_count < te.shape[0]:
            idx = rng.choice(te.shape[0], size=args.probe_count, replace=False)
            te = te[idx]
        return te

    probes_raw = {
        "en": sample_probe(args.en_split),
        "zh": sample_probe(args.zh_split),
        "prng": sample_probe(args.prng_split),
    }
    print("Probe shapes:", {k: v.shape for k, v in probes_raw.items()})

    models = {
        "en": load_model(args.en_model, specs["en"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
        "zh": load_model(args.zh_model, specs["zh"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
        "prng": load_model(args.prng_model, specs["prng"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
    }

    results = {}
    for model_name, model in models.items():
        print(f"--- Model {model_name} ---")
        m_shift = specs[model_name]["shift"]
        m_vocab = specs[model_name]["vocab"]
        results[model_name] = {}
        for probe_name, arr in probes_raw.items():
            # shift to model's index space and clip overflow
            shifted = (arr - m_shift)
            shifted = np.clip(shifted, 0, m_vocab - 1)
            loader = create_loader(shifted, args.batch_size)
            loss, acc = run_epoch(model, loader, criterion, device)
            print(f"Probe {probe_name}: loss={loss:.4f} acc={acc:.3f}")
            results[model_name][probe_name] = {"loss": loss, "acc": acc}
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
