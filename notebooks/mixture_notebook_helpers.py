"""Notebook helpers to run loss- and gradient-based mixture inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
ARTIFACTS = ROOT / "artifacts"
PYTHON_BIN = Path(sys.executable).resolve()
VENV_BIN = ROOT / ".venv" / "bin" / "python"
if VENV_BIN.exists():
    PYTHON_BIN = VENV_BIN


def _run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _plot_curve(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(df["true_en"], df["inferred_en"], marker="o", label="inferred")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    plt.xlabel("True EN proportion")
    plt.ylabel("Inferred EN proportion")
    plt.legend()
    plt.title(title)
    plt.tight_layout()


def run_mixture_analysis(
    mode: Literal["loss", "gradient"],
    probe_count: int = 500,
) -> pd.DataFrame:
    """Execute mixture inference script and return dataframe for plotting."""
    if mode == "loss":
        script = ROOT / "mixture_inference.py"
        out_json = ARTIFACTS / "mixture.json"
    elif mode == "gradient":
        script = ROOT / "mixture_inference_gradients.py"
        out_json = ARTIFACTS / "mixture_gradients.json"
    else:
        raise ValueError("mode must be 'loss' or 'gradient'")

    cmd = [
        str(PYTHON_BIN),
        str(script),
        "--en-split",
        str(ARTIFACTS / "splits" / "en-split.npz"),
        "--zh-split",
        str(ARTIFACTS / "splits" / "zh-split.npz"),
        "--en-model",
        str(ARTIFACTS / "en-split-lstm.pt"),
        "--zh-model",
        str(ARTIFACTS / "zh-split-lstm.pt"),
        "--probe-count",
        str(probe_count),
        "--out-json",
        str(out_json),
    ]
    _run_command(cmd)

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["rows"])
    _plot_curve(df, f"Mixture inference ({mode})")
    return df


def run_and_display(mode: Literal["loss", "gradient"], probe_count: int = 500) -> pd.DataFrame:
    """Convenience wrapper for Jupyter notebooks."""
    df = run_mixture_analysis(mode, probe_count)
    display(df)
    return df
