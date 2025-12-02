"""Prepare train/val/test splits for EN/ZH/PRNG with shared vocab (0-255)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_sequences(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["sequences"].astype(np.int64)


def split_90_10(
    arr: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.9 * n)
    train = arr[idx[:n_train]]
    val = arr[idx[n_train:]]
    return train, val


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]
    print(
        f"Saved {path} "
        f"keys={list(arrays.keys())} "
        f"shapes={[v.shape for v in arrays.values()]}"
    )


def prepare_and_save(
    name: str,
    train_path: Path,
    test_path: Path,
    out_dir: Path,
    global_min: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, shift, split 90/10, and save train/val/test NPZ for one dataset.
    """
    train_full = load_sequences(train_path) - global_min
    train, val = split_90_10(train_full, seed=seed)
    test = load_sequences(test_path) - global_min

    save_npz(
        out_dir / f"{name}-split.npz",
        train=train, val=val, test=test
    )
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for EN/ZH/PRNG."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for splits"
    )
    args = parser.parse_args()

    root = args.root
    seed = args.seed

    # load and shift values to start at zero (global min=1)
    global_min = 1

    out_dir = root / "splits"
    # TODO: add history versions of en and zh later
    en_train, en_val, en_ext = prepare_and_save(
        "en",
        root / "en-len16-n8192-rep1.npz",
        root / "en-len16-n8192-rep2.npz",
        out_dir,
        global_min,
        seed,
    )
    zh_train, zh_val, zh_ext = prepare_and_save(
        "zh",
        root / "zh-llm-len16-n8192-rep1.npz",
        root / "zh-llm-len16-n8192-rep2.npz",
        out_dir,
        global_min,
        seed,
    )
    prng_train, prng_val, prng_ext = prepare_and_save(
        "prng",
        root / "prng-len16-n8192-rep1.npz",
        root / "prng-len16-n8192-rep2.npz",
        out_dir,
        global_min,
        seed,
    )

    # log min/max after shift (should be >=0 and <=255)
    for name, arr in {
        "en_train_all": np.concatenate([en_train, en_val, en_ext]),
        "zh_train_all": np.concatenate([zh_train, zh_val, zh_ext]),
        "prng_all": np.concatenate([prng_train, prng_val, prng_ext]),
    }.items():
        print(f"{name}: min={arr.min()}, max={arr.max()}, shape={arr.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
