"""Utility snippet to plot gradient-norm confusion matrix inside a notebook."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_gradients import load_model, create_loader, grad_vector
from train_lstm_classifier import get_device, load_split

ARTIFACTS = ROOT / "artifacts"
SPLITS = {
    "en": ARTIFACTS / "splits" / "en-split.npz",
    "zh": ARTIFACTS / "splits" / "zh-split.npz",
    "prng": ARTIFACTS / "splits" / "prng-split.npz",
}
CKPTS = {
    "en": ARTIFACTS / "en-split-lstm.pt",
    "zh": ARTIFACTS / "zh-split-lstm.pt",
    "prng": ARTIFACTS / "prng-split-lstm.pt",
}


def collect_grad_matrix(probe_count: int = 200, batch_size: int = 128) -> pd.DataFrame:
    """Mirror probe_gradients.py but return a dataframe for visualization."""
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)

    specs: dict[str, dict[str, int]] = {}
    probes: dict[str, np.ndarray] = {}
    for name, split_path in SPLITS.items():
        tr, val, te = load_split(split_path)
        gmin = int(min(tr.min(), val.min(), te.min()))
        gmax = int(max(tr.max(), val.max(), te.max()))
        if probe_count and probe_count < te.shape[0]:
            te = te[rng.choice(te.shape[0], size=probe_count, replace=False)]
        specs[name] = {"shift": gmin, "vocab": gmax - gmin + 1}
        probes[name] = te

    models = {
        name: load_model(CKPTS[name], specs[name]["vocab"], 64, 128, 1, 0.1, device)
        for name in SPLITS
    }

    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        m_shift = specs[model_name]["shift"]
        m_vocab = specs[model_name]["vocab"]
        for probe_name, arr in probes.items():
            shifted = np.clip(arr - m_shift, 0, m_vocab - 1)
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
            rows.append(
                {
                    "model": model_name,
                    "probe": probe_name,
                    "avg_loss": total_loss / total_tokens,
                    "grad_norm": grad_norm,
                }
            )
    return pd.DataFrame(rows)


def plot_grad_confusion(df: pd.DataFrame) -> None:
    """Pivot dataframe and display a confusion-matrix-style heatmap."""
    order = ["en", "zh", "prng"]
    mat = (
        df.pivot(index="model", columns="probe", values="grad_norm")
        .reindex(order, axis=0)
        .reindex(order, axis=1)
    )

    out_png = ARTIFACTS / "gradient-confusion.png"
    plt.figure(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="magma_r")
    plt.title("Gradient Norm")
    plt.ylabel("Model")
    plt.xlabel("Probe")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = collect_grad_matrix()
    plot_grad_confusion(df)
