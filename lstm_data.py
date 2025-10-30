"""Utilities for loading and splitting sequence data for LSTM training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SequenceDataset:
    """Container holding train/validation/test splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _load_single_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "sequences" not in data:
            raise KeyError(f"expected 'sequences' key in {path}")
        seqs = data["sequences"]
    if seqs.ndim != 2:
        raise ValueError(
            f"sequences in {path} must be 2-dim, got shape {seqs.shape}")
    return seqs.astype(np.int64)


def load_sequences(dirs: Iterable[Path]) -> np.ndarray:
    """Load and concatenate sequences from a collection of NPZ directories."""
    seqs: list[np.ndarray] = []
    for dir in dirs:
        if not dir.is_dir():
            raise NotADirectoryError(f"{dir} is not a directory")
        files = sorted(dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no NPZ files found in {dir}")
        for file in files:
            seqs.append(_load_single_npz(file))
    if not seqs:
        raise ValueError("no sequences were loaded")
    merged_seqs = np.concatenate(seqs, axis=0)
    return merged_seqs


def map_to_class_indices(seqs: np.ndarray) -> np.ndarray:
    """Shift integer sequences so the smallest value becomes zero."""
    min_value = seqs.min()
    shifted_seqs = seqs - min_value
    return shifted_seqs


def train_val_test_split(
    seqs: np.ndarray,
    splits: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int | None = 42,
) -> SequenceDataset:
    """Split sequences into train/validation/test arrays."""
    if not np.isclose(sum(splits), 1.0):
        raise ValueError("Splits must sum to 1.0")

    num_samples = seqs.shape[0]
    indices = np.arange(num_samples)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_end = int(splits[0] * num_samples)
    val_end = train_end + int(splits[1] * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return SequenceDataset(
        train=seqs[train_idx],
        val=seqs[val_idx],
        test=seqs[test_idx]
    )
