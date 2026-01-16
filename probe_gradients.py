"""Compute gradient-based probes for EN/ZH/PRNG models on probe sets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_lstm_classifier import LstmModel, load_split, get_device


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


def shift_with_min(arr: np.ndarray, gmin: int) -> np.ndarray:
    return (arr - gmin).astype(np.int64)


def load_model(model_path: Path, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float, device: torch.device) -> LstmModel:
    model = LstmModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    state = torch.load(model_path, map_location=device)
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
    return model


def grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name.startswith("embedding") or name.startswith("lstm"):
            grads.append(p.grad.view(-1))
    if not grads:
        return torch.tensor([], device=next(model.parameters()).device)
    return torch.cat(grads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gradient probes for EN/ZH/PRNG models on probe sets.")
    parser.add_argument("--en-split", type=Path, default=Path("artifacts/splits/en-split.npz"))
    parser.add_argument("--zh-split", type=Path, default=Path("artifacts/splits/zh-split.npz"))
    parser.add_argument("--prng-split", type=Path, default=Path("artifacts/splits/prng-split.npz"))
    parser.add_argument("--en-model", type=Path, default=Path("artifacts/en-split-lstm.pt"))
    parser.add_argument("--zh-model", type=Path, default=Path("artifacts/zh-split-lstm.pt"))
    parser.add_argument("--prng-model", type=Path, default=Path("artifacts/prng-split-lstm.pt"))
    parser.add_argument("--probe-count", type=int, default=200, help="Sequences per probe set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args(argv)

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)

    def dataset_stats(split_path: Path):
        tr, val, te = load_split(split_path)
        gmin = int(min(tr.min(), val.min(), te.min()))
        gmax = int(max(tr.max(), val.max(), te.max()))
        if args.probe_count and args.probe_count < te.shape[0]:
            idx = rng.choice(te.shape[0], size=args.probe_count, replace=False)
            te = te[idx]
        return tr, val, te, gmin, gmax

    datasets = {
        "en": dataset_stats(args.en_split),
        "zh": dataset_stats(args.zh_split),
        "prng": dataset_stats(args.prng_split),
    }

    specs = {}
    probes = {}
    for name, (tr, val, te, gmin, gmax) in datasets.items():
        vocab = gmax - gmin + 1
        specs[name] = {"shift": gmin, "vocab": vocab}
        probes[name] = te
        print(f"{name}: shift={gmin}, vocab={vocab}, probe shape={te.shape}")

    models = {
        "en": load_model(args.en_model, specs["en"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
        "zh": load_model(args.zh_model, specs["zh"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
        "prng": load_model(args.prng_model, specs["prng"]["vocab"], args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device),
    }

    for model_name, model in models.items():
        print(f"--- Model {model_name} ---")
        m_shift = specs[model_name]["shift"]
        m_vocab = specs[model_name]["vocab"]
        for probe_name, arr in probes.items():
            # shift/crop probe to model space
            shifted = arr - m_shift
            shifted = np.clip(shifted, 0, m_vocab - 1)
            loader = create_loader(shifted, args.batch_size)
            model.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_tokens = 0
            # one full pass over probe to accumulate gradients
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
            gvec = grad_vector(model)
            norm = gvec.norm().item() if gvec.numel() > 0 else 0.0
            print(f"Probe {probe_name}: avg_loss={total_loss/total_tokens:.4f}, grad_norm={norm:.4f}, grad_dim={gvec.numel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
