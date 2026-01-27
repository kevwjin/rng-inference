"""Compute probe losses and plot a heatmap (magma_r) for EN/ZH/PRNG."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn

from probe_losses import create_loader, run_epoch, load_model
from train_lstm_classifier import get_device, load_split


def collect_loss_matrix(
    en_split: Path,
    zh_split: Path,
    prng_split: Path,
    en_model: Path,
    zh_model: Path,
    prng_model: Path,
    probe_count: int = 500,
    batch_size: int = 128,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> pd.DataFrame:
    """Run cross-probe losses and return a dataframe."""
    device = get_device()
    rng = np.random.default_rng(42)
    criterion = nn.CrossEntropyLoss()

    splits = {"en": en_split, "zh": zh_split, "prng": prng_split}
    ckpts = {"en": en_model, "zh": zh_model, "prng": prng_model}

    specs: dict[str, dict[str, int]] = {}
    probes: dict[str, np.ndarray] = {}
    for name, split_path in splits.items():
        tr, val, te = load_split(split_path)
        gmin = int(min(tr.min(), val.min(), te.min()))
        gmax = int(max(tr.max(), val.max(), te.max()))
        if probe_count and probe_count < te.shape[0]:
            te = te[rng.choice(te.shape[0], size=probe_count, replace=False)]
        specs[name] = {"shift": gmin, "vocab": gmax - gmin + 1}
        probes[name] = te

    models = {
        name: load_model(
            ckpts[name],
            specs[name]["vocab"],
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            device,
        )
        for name in splits
    }

    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        m_shift = specs[model_name]["shift"]
        m_vocab = specs[model_name]["vocab"]
        for probe_name, arr in probes.items():
            shifted = np.clip(arr - m_shift, 0, m_vocab - 1)
            loader = create_loader(shifted, batch_size)
            loss, acc = run_epoch(model, loader, criterion, device)
            rows.append(
                {
                    "model": model_name,
                    "probe": probe_name,
                    "loss": loss,
                    "acc": acc,
                }
            )
    return pd.DataFrame(rows)


def plot_loss_confusion(df: pd.DataFrame, out_png: Path) -> None:
    """Pivot dataframe and display/save a confusion-matrix-style heatmap."""
    order = ["en", "zh", "prng"]
    mat = (
        df.pivot(index="model", columns="probe", values="loss")
        .reindex(order, axis=0)
        .reindex(order, axis=1)
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="magma_r")
    plt.title("Probe Loss")
    plt.ylabel("Model")
    plt.xlabel("Probe")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot probe-loss confusion matrix.")
    parser.add_argument("--en-split", type=Path, default=Path("artifacts/splits/en-split.npz"))
    parser.add_argument("--zh-split", type=Path, default=Path("artifacts/splits/zh-split.npz"))
    parser.add_argument("--prng-split", type=Path, default=Path("artifacts/splits/prng-split.npz"))
    parser.add_argument("--en-model", type=Path, default=Path("artifacts/en-split-lstm.pt"))
    parser.add_argument("--zh-model", type=Path, default=Path("artifacts/zh-split-lstm.pt"))
    parser.add_argument("--prng-model", type=Path, default=Path("artifacts/prng-split-lstm.pt"))
    parser.add_argument("--probe-count", type=int, default=500, help="Sequences per probe set (0 or large=full set)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out-png", type=Path, default=Path("artifacts/probe-loss-confusion.png"))
    args = parser.parse_args(argv)

    df = collect_loss_matrix(
        en_split=args.en_split,
        zh_split=args.zh_split,
        prng_split=args.prng_split,
        en_model=args.en_model,
        zh_model=args.zh_model,
        prng_model=args.prng_model,
        probe_count=args.probe_count,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    plot_loss_confusion(df, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
