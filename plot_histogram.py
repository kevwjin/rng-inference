"""Generate histograms for every samples/<lang-len> directory."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SAMPLES_ROOT = Path("samples")


def iter_sequences(npz_paths: Iterable[Path]) -> Iterable[np.ndarray]:
    """Yield flattened integer arrays from each NPZ file."""
    for path in npz_paths:
        with np.load(path) as data:
            if "sequences" not in data:
                raise KeyError(f"NPZ file {path} missing 'sequences' array")
            yield data["sequences"].reshape(-1)


def build_histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram counts and bin edges covering the value range."""
    min_value = int(values.min())
    max_value = int(values.max())
    bins = max_value - min_value + 1
    return np.histogram(values, bins=bins, range=(min_value, max_value + 1))


def plot_histogram(counts: np.ndarray, edges: np.ndarray, output: Path) -> None:
    """Render and save the histogram figure to the specified path."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(edges[:-1], counts, width=0.9, align="edge", edgecolor="black")
    ax.set_xlabel("Integer value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of generated integers")
    ax.set_xlim(edges[0], edges[-1])
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def process_directory(directory: Path) -> None:
    npz_files = sorted(directory.glob("*.npz"))
    if not npz_files:
        return

    flattened = np.concatenate(list(iter_sequences(npz_files)))
    counts, edges = build_histogram(flattened)

    output_path = directory / "histogram.png"
    plot_histogram(counts, edges, output_path)
    print(f"Saved {output_path}")


def main() -> int:
    if not SAMPLES_ROOT.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {SAMPLES_ROOT}")

    subdirs = sorted(path for path in SAMPLES_ROOT.iterdir() if path.is_dir())
    if not subdirs:
        print("No subdirectories found under samples/ to process.")
        return 0

    for directory in subdirs:
        process_directory(directory)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
