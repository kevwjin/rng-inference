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

    # TODO: add history versions of en and zh later
    files = {
        "en_train": root / "en-len16-n8192-rep1.npz",
        "en_test": root / "en-len16-n8192-rep2.npz",
        "zh_train": root / "zh-llm-len16-n8192-rep1.npz",
        "zh_test": root / "zh-llm-len16-n8192-rep2.npz",
        "prng_train": root / "prng-len16-n8192-rep1.npz",
        "prng_test": root / "prng-len16-n8192-rep2.npz",
    }

    # load and shift values to start at zero (global min=1)
    global_min = 1
    datasets = {}
    for name, path in files.items():
        datasets[name] = load_sequences(path) - global_min

    # EN: split rep1 90/10; use rep2 as external test
    en_train, en_val = split_90_10(datasets["en_train"], seed=seed)
    en_ext = datasets["en_test"]

    # ZH: only rep1 available; split 90/10 (no external test yet)
    zh_train, zh_val = split_90_10(datasets["zh_train"], seed=seed)
    zh_ext = np.empty((0, *zh_val.shape[1:]), dtype=zh_val.dtype)

    # PRNG: split rep1 90/10; use rep2 as external test
    prng_train, prng_val = split_90_10(datasets["prng_train"], seed=seed)
    prng_ext = datasets["prng_test"]

    out_dir = root / "splits"
    save_npz(
        out_dir / "en-split.npz",
        train=en_train,
        val=en_val,
        test=en_ext
    )
    save_npz(
        out_dir / "zh-split.npz",
        train=zh_train,
        val=zh_val,
        test=zh_ext
    )
    save_npz(
        out_dir / "prng-split.npz",
        train=prng_train,
        val=prng_val,
        test=prng_ext
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
