"""Estimate EN proportion in EN/ZH mixes via gradient norms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_lstm_classifier import LstmModel, get_device, load_split


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
    model.eval()
    return model


def grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith("embedding") or name.startswith("lstm"):
            grads.append(param.grad.view(-1))
    if not grads:
        return torch.tensor([], device=next(model.parameters()).device)
    return torch.cat(grads)


def compute_shift_and_vocab(splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int]:
    tr, val, te = splits
    gmin = int(min(tr.min(), val.min(), te.min()))
    gmax = int(max(tr.max(), val.max(), te.max()))
    return gmin, gmax - gmin + 1


def sample_mix(
    en_data: np.ndarray,
    zh_data: np.ndarray,
    probe_count: int,
    p_en: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = probe_count
    n_en = int(p_en * n)
    n_zh = n - n_en
    seq_len = en_data.shape[1]
    parts = []
    if n_en > 0:
        idx = rng.choice(en_data.shape[0], size=n_en, replace=False)
        parts.append(en_data[idx])
    if n_zh > 0:
        idx = rng.choice(zh_data.shape[0], size=n_zh, replace=False)
        parts.append(zh_data[idx])
    if not parts:
        return np.empty((0, seq_len), dtype=np.int64)
    mix = np.vstack(parts).astype(np.int64)
    rng.shuffle(mix, axis=0)
    return mix


def grad_metrics(
    model: nn.Module,
    mix_raw: np.ndarray,
    shift: int,
    vocab: int,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    if mix_raw.size == 0:
        return 0.0, 0.0
    shifted = np.clip(mix_raw - shift, 0, vocab - 1)
    loader = create_loader(shifted, batch_size)
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    gvec = grad_vector(model)
    grad_norm = gvec.norm().item() if gvec.numel() else 0.0
    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    return avg_loss, grad_norm


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate EN proportion in EN/ZH mixes via gradient norms.")
    parser.add_argument("--en-split", type=Path, default=Path("artifacts/splits/en-split.npz"))
    parser.add_argument("--zh-split", type=Path, default=Path("artifacts/splits/zh-split.npz"))
    parser.add_argument("--en-model", type=Path, default=Path("artifacts/en-split-lstm.pt"))
    parser.add_argument("--zh-model", type=Path, default=Path("artifacts/zh-split-lstm.pt"))
    parser.add_argument("--probe-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out-json", type=Path, help="Optional JSON path to save mix results")
    args = parser.parse_args(argv)

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)

    en_splits = load_split(args.en_split)
    zh_splits = load_split(args.zh_split)
    en_min, en_vocab = compute_shift_and_vocab(en_splits)
    zh_min, zh_vocab = compute_shift_and_vocab(zh_splits)
    print(f"EN shift={en_min}, vocab={en_vocab}; ZH shift={zh_min}, vocab={zh_vocab}")

    en_model = load_model(args.en_model, en_vocab, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device)
    zh_model = load_model(args.zh_model, zh_vocab, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device)

    en_te_raw = en_splits[2]
    zh_te_raw = zh_splits[2]
    proportions = [0.0, 0.25, 0.5, 0.75, 1.0]

    rows = []
    for p in proportions:
        mix_raw = sample_mix(en_te_raw, zh_te_raw, args.probe_count, p, rng)
        loss_en, grad_en = grad_metrics(en_model, mix_raw, en_min, en_vocab, args.batch_size, device, criterion)
        loss_zh, grad_zh = grad_metrics(zh_model, mix_raw, zh_min, zh_vocab, args.batch_size, device, criterion)
        rows.append(
            {
                "true_en": p,
                "loss_en": loss_en,
                "loss_zh": loss_zh,
                "grad_en": grad_en,
                "grad_zh": grad_zh,
            }
        )

    y = np.array([row["true_en"] for row in rows])
    X = np.array([[row["grad_en"], row["grad_zh"]] for row in rows])
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    print(f"Fitted coeffs (bias, b_grad_en, c_grad_zh): {coeffs}")

    print("prop_en, grad_en, grad_zh, loss_en, loss_zh, inferred_en_prop")
    for row in rows:
        inp = np.array([1.0, row["grad_en"], row["grad_zh"]])
        pred = float(np.dot(inp, coeffs))
        pred = max(0.0, min(1.0, pred))
        print(
            f"{row['true_en']:.2f}, {row['grad_en']:.4f}, {row['grad_zh']:.4f}, "
            f"{row['loss_en']:.4f}, {row['loss_zh']:.4f}, {pred:.3f}"
        )
        row["inferred_en"] = pred

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"coeffs": coeffs.tolist(), "rows": rows}, f, indent=2)
        print(f"Saved mixture results to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
