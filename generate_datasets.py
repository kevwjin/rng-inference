#!/usr/bin/env python
"""Generate ARS and NARS datasets for en/zh LLM prompts and RNG baselines."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

ARTIFACT_ROOT = Path("artifacts")


def run_command(args: list[str]) -> None:
    proc = subprocess.run(args, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def run_llm(lang: str, seq_len: int, num_seq: int, output_path: Path) -> Path:
    args = [
        "python",
        "sample_llm.py",
        "--prompt-language",
        lang,
        "--sequence-length",
        str(seq_len),
        "--num-sequences",
        str(num_seq),
        "--save-npz",
        "--output-path",
        str(output_path),
    ]
    run_command(args)
    return output_path


def run_rng(seq_len: int, num_seq: int, seed: int | None, output_path: Path) -> Path:
    args = [
        "python",
        "sample_rng.py",
        "--sequence-length",
        str(seq_len),
        "--num-sequences",
        str(num_seq),
        "--save-npz",
        "--output-path",
        str(output_path),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])
    run_command(args)
    return output_path


def reshape_npz(src: Path, target_len: int, dst: Path) -> None:
    data = np.load(src)["sequences"]
    flat = data.reshape(-1)
    if flat.size % target_len != 0:
        raise RuntimeError(
            f"Total elements {flat.size} not divisible by {target_len} from {src}"
        )
    reshaped = flat.reshape(-1, target_len)
    np.savez(dst, sequences=reshaped.astype("int64"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate ARS/NARS datasets for en/zh LLM prompts and RNG"
    )
    parser.add_argument(
        "--total-numbers",
        type=int,
        default=131072,
        help="Total integers per dataset (default: 131072; 2^17, divisible by 32)",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[16, 32],
        help="Sequence lengths to generate (default: 16 32)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of repeats per condition (default: 2)",
    )
    args = parser.parse_args(argv)

    sources = ["en", "zh", "rng"]

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for src in sources:
        for seq_len in args.lengths:
            if args.total_numbers % seq_len != 0:
                raise RuntimeError(
                    f"Total numbers ({args.total_numbers}) not divisible by "
                    f"sequence length {seq_len}"
                )
            num_seq = args.total_numbers // seq_len

            for rep in range(1, args.repeats + 1):
                # ARS: whole sequence per call
                ars_target = (
                    ARTIFACT_ROOT / f"{src}-ars-len{seq_len}-n{num_seq}-rep{rep}.npz"
                )
                if src == "rng":
                    run_rng(seq_len, num_seq, None, ars_target)
                else:
                    run_llm(src, seq_len, num_seq, ars_target)
                if not ars_target.exists():
                    raise RuntimeError(f"Expected ARS artifact missing: {ars_target}")
                print(f"Saved ARS -> {ars_target}")

                # NARS: single-number calls reshaped to target length
                nars_target = (
                    ARTIFACT_ROOT / f"{src}-nars-len{seq_len}-n{num_seq}-rep{rep}.npz"
                )
                nars_src = ARTIFACT_ROOT / f"{src}-nars-temp-len1-rep{rep}.npz"
                if src == "rng":
                    run_rng(1, args.total_numbers, None, nars_src)
                else:
                    run_llm(src, 1, args.total_numbers, nars_src)

                if not nars_src.exists():
                    raise RuntimeError(f"Expected NARS artifact missing: {nars_src}")
                reshape_npz(nars_src, seq_len, nars_target)
                nars_src.unlink()
                print(f"Saved NARS -> {nars_target}")

    print("Done. Artifacts in artifacts/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
