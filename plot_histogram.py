"""Generate histograms for every artifacts/*.npz file."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_ROOT = Path("artifacts")


def load_flattened_sequences(npz_path: Path) -> np.ndarray:
    """Load and flatten the sequences array from an NPZ artifact."""
    with np.load(npz_path) as data:
        if "sequences" not in data:
            raise KeyError(f"NPZ file {npz_path} missing 'sequences' array")
        sequences = data["sequences"]
    return sequences.reshape(-1)


def build_histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram counts and bin edges covering the value range."""
    min_value = int(values.min())
    max_value = int(values.max())
    bins = max_value - min_value + 1
    return np.histogram(values, bins=bins, range=(min_value, max_value + 1))


def plot_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    output: Path
) -> None:
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


def process_file(npz_path: Path) -> None:
    flattened = load_flattened_sequences(npz_path)
    counts, edges = build_histogram(flattened)

    output_path = npz_path.with_name(f"{npz_path.stem}-hist.png")
    plot_histogram(counts, edges, output_path)
    print(f"Saved {output_path}")


def main() -> int:
    if not ARTIFACT_ROOT.is_dir():
        raise \
            FileNotFoundError(f"Artifact directory not found: {ARTIFACT_ROOT}")

    npz_files = sorted(ARTIFACT_ROOT.glob("*.npz"))
    if not npz_files:
        print("No NPZ artifacts found under artifacts/ to process.")
        return 0

    for npz_path in npz_files:
        process_file(npz_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
