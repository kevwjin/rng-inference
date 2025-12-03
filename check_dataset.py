"""Integrity checks and simple diagnostics for NPZ sequence datasets."""
from __future__ import annotations

from scipy.stats import chisquare

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ChiSquareResult:
    stat: float
    dof: int
    p_value: float | None


@dataclass
class PositionSummary:
    index: int
    unique_values: int
    max_count: int
    max_value: int
    zero_bins: int
    min_nonzero_count: int
    chi_square_uniform: ChiSquareResult
    chi_square_vs_global: ChiSquareResult


@dataclass
class DatasetSummary:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    min_value: int
    max_value: int
    duplicate_rows: int
    duplicate_entries: int
    unique_rows: int
    histogram_z_outliers: list[tuple[int, int, float]]
    chi_square_uniform: ChiSquareResult
    position_summaries: list[PositionSummary]


def load_sequences(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with np.load(path) as data:
        seqs = data["sequences"]
    return seqs.astype(np.int64)


def _chi_square(counts: np.ndarray, expected: np.ndarray) -> ChiSquareResult:
    """Compute chi-square stat and p-value using SciPy's chisquare."""
    mask = expected > 0
    if not np.any(mask):
        return ChiSquareResult(stat=0.0, dof=0, p_value=None)
    obs = counts[mask]
    exp = expected[mask]
    dof = int(len(obs) - 1)
    if dof <= 0:
        return ChiSquareResult(stat=0.0, dof=dof, p_value=None)
    stat, p = chisquare(f_obs=obs, f_exp=exp)
    return ChiSquareResult(stat=float(stat), dof=dof, p_value=float(p))


def _histogram(
    seqs: np.ndarray,
    min_value: int | None = None,
    max_value: int | None = None
) -> tuple[np.ndarray, int, int]:
    flat = seqs.reshape(-1)
    min_v = int(flat.min()) if min_value is None else int(min_value)
    max_v = int(flat.max()) if max_value is None else int(max_value)
    num_bins = max_v - min_v + 1
    counts = np.bincount(flat - min_v, minlength=num_bins)
    return counts, min_v, max_v


def _z_outliers(
    counts: np.ndarray, min_value: int, threshold: float = 3.0
) -> list[tuple[int, int, float]]:
    total = counts.sum()
    num_bins = len(counts)
    if num_bins == 0 or total == 0:
        return []
    expected = total / num_bins
    std = expected ** 0.5
    if std == 0:
        return []
    z_scores = (counts - expected) / std
    outlier_indices = np.where(np.abs(z_scores) >= threshold)[0]
    result: list[tuple[int, int, float]] = []
    for idx in outlier_indices:
        value = min_value + int(idx)
        result.append((value, int(counts[idx]), float(z_scores[idx])))
    return result


def _position_histograms(
    seqs: np.ndarray, min_value: int, max_value: int
) -> np.ndarray:
    num_bins = max_value - min_value + 1
    seq_len = seqs.shape[1]
    hists = np.zeros((seq_len, num_bins), dtype=np.int64)
    offset = -min_value
    for idx in range(seq_len):
        values = seqs[:, idx] + offset
        hists[idx] = np.bincount(values, minlength=num_bins)
    return hists


def _summarize_positions(
    position_hists: np.ndarray,
    global_hist: np.ndarray,
    min_value: int,
) -> list[PositionSummary]:
    summaries: list[PositionSummary] = []
    total_global = global_hist.sum()
    global_probs = None
    if total_global:
        global_probs = global_hist / total_global
    else:
        global_probs = np.zeros_like(global_hist, dtype=float)
    num_bins = global_hist.shape[0]

    for idx, hist in enumerate(position_hists):
        total = hist.sum()
        if total == 0:
            continue
        expected_uniform = np.full(num_bins, total / num_bins, dtype=float)
        expected_global = global_probs * total

        nonzero = hist[hist > 0]
        min_nonzero = int(nonzero.min()) if nonzero.size else 0

        summary = PositionSummary(
            index=idx,
            unique_values=int(np.count_nonzero(hist)),
            max_count=int(hist.max()),
            max_value=int(min_value + hist.argmax()),
            zero_bins=int(np.count_nonzero(hist == 0)),
            min_nonzero_count=min_nonzero,
            chi_square_uniform=_chi_square(hist, expected_uniform),
            chi_square_vs_global=_chi_square(hist, expected_global),
        )
        summaries.append(summary)
    return summaries


def summarize_dataset(path: Path) -> DatasetSummary:
    seqs = load_sequences(path)
    hist, min_value, max_value = _histogram(seqs)
    total = hist.sum()  # total number of datapoints in hist
    num_bins = len(hist)
    expected_uniform = np.full(num_bins, total / num_bins)

    chi_sq_uniform = _chi_square(hist, expected_uniform)
    z_outliers = _z_outliers(hist, min_value)

    unique_rows, counts = np.unique(seqs, axis=0, return_counts=True)
    duplicate_entries = int(counts[counts > 1].sum())
    duplicate_rows = int(np.count_nonzero(counts > 1))

    position_hists = _position_histograms(seqs, min_value, max_value)
    position_summaries = _summarize_positions(position_hists, hist, min_value)

    return DatasetSummary(
        path=path,
        shape=seqs.shape,
        dtype=str(seqs.dtype),
        min_value=min_value,
        max_value=max_value,
        duplicate_rows=duplicate_rows,
        duplicate_entries=duplicate_entries,
        unique_rows=int(len(unique_rows)),
        histogram_z_outliers=z_outliers,
        chi_square_uniform=chi_sq_uniform,
        position_summaries=position_summaries,
    )


def summarize_many(paths: Iterable[Path]) -> list[DatasetSummary]:
    return [summarize_dataset(Path(path)) for path in paths]


def _format_chi(res: ChiSquareResult) -> str:
    p_str = f", p={res.p_value:.3e}" if res.p_value is not None else ""
    return f"chi2={res.stat:.2f}, dof={res.dof}{p_str}"


def print_report(summary: DatasetSummary) -> None:
    print(f"File: {summary.path}")
    print(f"  shape={summary.shape}, dtype={summary.dtype}")
    print(f"  min={summary.min_value}, max={summary.max_value}")
    print(f"  unique rows={summary.unique_rows}, "
          f"duplicate rows={summary.duplicate_rows}, "
          f"duplicate entries={summary.duplicate_entries}"
    )
    print(f"  global uniform test: {_format_chi(summary.chi_square_uniform)}")
    if summary.histogram_z_outliers:
        print("  value frequency outliers (value, count, z):")
        for value, count, z in summary.histogram_z_outliers:
            print(f"    {value}: count={count}, z={z:.2f}")
    else:
        print("  value frequency outliers: none")

    print("  per-position summary:")
    for pos in summary.position_summaries:
        print(
            f"    pos {pos.index}: unique={pos.unique_values}, "
            f"max_count={pos.max_count} (value={pos.max_value}), "
            f"zero_bins={pos.zero_bins}, "
            f"min_nonzero={pos.min_nonzero_count}, "
            f"uniform[{_format_chi(pos.chi_square_uniform)}], "
            f"vs_global[{_format_chi(pos.chi_square_vs_global)}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run integrity checks on NPZ sequence datasets.")
    parser.add_argument("paths", nargs="+", help="NPZ files to summarize")
    args = parser.parse_args(argv)

    for report in summarize_many(args.paths):
        print_report(report)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
