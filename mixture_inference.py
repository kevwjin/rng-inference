"""Estimate EN proportion in EN/ZH mixes via model losses."""
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


def load_split(path: Path):
    with np.load(path) as data:
        return data["train"], data["val"], data["test"]


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate EN proportion in EN/ZH mixes via losses.")
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

    # Load splits and compute shift/vocab per dataset (from train/val/test)
    en_tr, en_val, en_te = load_split(args.en_split)
    zh_tr, zh_val, zh_te = load_split(args.zh_split)
    en_min, en_max = int(en_tr.min()), int(en_tr.max())
    zh_min, zh_max = int(zh_tr.min()), int(zh_tr.max())
    en_vocab = en_max - en_min + 1
    zh_vocab = zh_max - zh_min + 1
    print(f"EN shift={en_min}, vocab={en_vocab}; ZH shift={zh_min}, vocab={zh_vocab}")

    # Shift datasets to their own index spaces
    en_te = (en_te - en_min).astype(np.int64)
    zh_te = (zh_te - zh_min).astype(np.int64)

    en_model = load_model(args.en_model, en_vocab, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device)
    zh_model = load_model(args.zh_model, zh_vocab, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, device)

    proportions = [0.0, 0.25, 0.5, 0.75, 1.0]

    def sample_mix(p_en: float) -> tuple[np.ndarray, np.ndarray]:
        n = args.probe_count
        n_en = int(p_en * n)
        n_zh = n - n_en
        en_idx = rng.choice(en_te.shape[0], size=n_en, replace=False) if n_en > 0 else []
        zh_idx = rng.choice(zh_te.shape[0], size=n_zh, replace=False) if n_zh > 0 else []
        parts_en = en_te[en_idx] if n_en > 0 else np.empty((0, en_te.shape[1]), dtype=np.int64)
        parts_zh = zh_te[zh_idx] if n_zh > 0 else np.empty((0, zh_te.shape[1]), dtype=np.int64)
        # For scoring with each model, shift to its vocab space (already done), then clip to vocab-1
        mix_for_en = np.vstack([parts_en, np.clip(parts_zh + (en_min - zh_min), 0, en_vocab - 1)]) if parts_zh.size else parts_en
        mix_for_zh = np.vstack([np.clip(parts_en + (zh_min - en_min), 0, zh_vocab - 1), parts_zh]) if parts_en.size else parts_zh
        return mix_for_en, mix_for_zh

    # Collect features for simple regression (use loss_en, loss_zh as features)
    feats = []
    for p in proportions:
        mix_en, mix_zh = sample_mix(p)
        loader_en = create_loader(mix_en, args.batch_size)
        loader_zh = create_loader(mix_zh, args.batch_size)
        loss_en, acc_en = run_epoch(en_model, loader_en, criterion, device)
        loss_zh, acc_zh = run_epoch(zh_model, loader_zh, criterion, device)
        feats.append((p, loss_en, loss_zh, acc_en, acc_zh))

    # Fit linear regressor: p_en = a + b*loss_en + c*loss_zh
    y = np.array([f[0] for f in feats])
    X = np.array([[f[1], f[2]] for f in feats])
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    print(f"Fitted coeffs (bias, b_loss_en, c_loss_zh): {coeffs}")

    print("prop_en, loss_en, loss_zh, acc_en, acc_zh, inferred_en_prop")
    mix_rows = []
    for p, loss_en, loss_zh, acc_en, acc_zh in feats:
        pred = float(np.dot(np.array([1.0, loss_en, loss_zh]), coeffs))
        pred = max(0.0, min(1.0, pred))
        print(f"{p:.2f}, {loss_en:.4f}, {loss_zh:.4f}, {acc_en:.3f}, {acc_zh:.3f}, {pred:.3f}")
        mix_rows.append({"true_en": p, "loss_en": loss_en, "loss_zh": loss_zh, "acc_en": acc_en, "acc_zh": acc_zh, "inferred_en": pred})

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"coeffs": coeffs.tolist(), "rows": mix_rows}, f, indent=2)
        print(f"Saved mixture results to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
