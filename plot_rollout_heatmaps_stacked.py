"""Stacked rollout heatmaps with a single shared colorbar."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
EN_ROLLOUTS = ROOT / "rollouts" / "en"

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.titlesize": 20,
        "legend.fontsize": 20,
    }
)


def sort_key(p: Path) -> int:
    """Sort by length encoded as en_len{N}_rollouts*.json."""
    stem = p.stem
    for part in stem.split("_"):
        if part.startswith("len"):
            try:
                return int(part.removeprefix("len"))
            except ValueError:
                return 0
    return 0


def parse_parts(path: Path) -> Dict[str, Optional[int]]:
    """Parse language, length, and rollouts from filename."""
    stem = path.stem
    parts = stem.split("_")
    lang = parts[0].upper() if parts else stem
    length = None
    rollouts = None
    for part in parts:
        if part.startswith("len"):
            try:
                length = int(part.removeprefix("len"))
            except ValueError:
                pass
        if part.startswith("rollouts"):
            try:
                rollouts = int(part.removeprefix("rollouts"))
            except ValueError:
                pass
    return {"lang": lang, "length": length, "rollouts": rollouts}


def load_matrices(paths: List[Path]) -> List[Dict]:
    """Load and normalize rollout matrices."""
    all_ints = set()
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = parse_parts(path)
        steps = data["per_step_probs"]
        ints = sorted(int(i) for i in steps[0].keys())
        all_ints.update(ints)
        mat = np.array([[step[str(i)] for i in ints] for step in steps], dtype=np.float32)
        mat_rel = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)
        records.append(
            {
                "path": path,
                "mat_rel": mat_rel,
                "ints": ints,
                "length": meta["length"],
                "rollouts": meta["rollouts"],
            }
        )
    # finalize global int axis
    global_ints = sorted(all_ints)
    int_index = {v: i for i, v in enumerate(global_ints)}
    for rec in records:
        mat_rel = rec["mat_rel"]
        padded = np.full((mat_rel.shape[0], len(global_ints)), np.nan, dtype=np.float32)
        for local_j, value in enumerate(rec["ints"]):
            padded[:, int_index[value]] = mat_rel[:, local_j]
        rec["mat_rel"] = padded
    return records, global_ints


def main() -> None:
    paths = sorted(EN_ROLLOUTS.glob("*.json"), key=sort_key)
    if not paths:
        raise SystemExit("No rollout JSON files found.")

    records, global_ints = load_matrices(paths)

    max_steps = max(rec["mat_rel"].shape[0] for rec in records)
    global_min_int, global_max_int = min(global_ints), max(global_ints)

    # Compute shared color scale.
    all_vals = np.concatenate([np.ravel(rec["mat_rel"][~np.isnan(rec["mat_rel"])]) for rec in records])
    vmax = np.percentile(all_vals, 99.5)
    norm = colors.PowerNorm(gamma=0.2, vmin=0, vmax=vmax)

    lengths = [rec["length"] for rec in records]
    rollouts = records[0]["rollouts"] if records and records[0]["rollouts"] else None

    # Layout sizing
    per_step_width = 0.8  # inches per step column
    cb_width_in = 0.5     # inches reserved for colorbar
    per_int_height = 0.08
    subplot_height = max(2.5, min(8.0, per_int_height * len(global_ints)))
    fig_height = subplot_height * len(records)
    fig_width = per_step_width * max_steps + cb_width_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=len(records),
        ncols=2,
        width_ratios=[per_step_width * max_steps, cb_width_in],
        height_ratios=[subplot_height] * len(records),
        hspace=0.05,
        wspace=0.05,
    )

    shared_cax = fig.add_subplot(gs[:, 1])
    im_for_cbar = None

    extent = [0.5, max_steps + 0.5, global_min_int - 0.5, global_max_int + 0.5]
    for idx, rec in enumerate(records):
        ax = fig.add_subplot(gs[idx, 0])
        mat_rel = rec["mat_rel"]
        padded = np.full((max_steps, mat_rel.shape[1]), np.nan, dtype=np.float32)
        padded[: mat_rel.shape[0], :] = mat_rel

        im = ax.imshow(
            padded.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=norm,
        )
        if im_for_cbar is None:
            im_for_cbar = im

        ax.set_xlim(0.5, max_steps + 0.5)
        ax.set_ylim(global_min_int - 0.5, global_max_int + 0.5)
        ax.set_ylabel("Integer")
        ax.set_xticks(range(1, max_steps + 1))
        if idx == len(records) - 1:
            ax.set_xlabel("Step")
        else:
            ax.set_xticklabels([])

    cbar = fig.colorbar(im_for_cbar, cax=shared_cax, label="Normalized probability")
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Normalized probability", fontsize=20)

    fig.suptitle("Integer Probabilities (EN)", fontsize=24, y=0.893)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("rollout-heatmaps-stacked.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
