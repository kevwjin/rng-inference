"""Notebook helpers for membership inference plots (loss + gradient)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mi_eval import evaluate_membership


def _metrics_dataframe(summary: dict) -> pd.DataFrame:
    rows = []
    for idx, run in enumerate(summary["runs"]):
        rows.append(
            {
                "run": idx,
                "n_members": run["n_mem"],
                "n_nonmembers": run["n_non"],
                "auc": run["auc"],
                "average_precision": run["average_precision"],
                "tpr_at_fpr": run["tpr_at_fpr"],
            }
        )
    return pd.DataFrame(rows)


def _plot_hist(run: dict, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(run["scores_mem"], bins=40, alpha=0.6, label="members")
    plt.hist(run["scores_non"], bins=40, alpha=0.6, label="non-members")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def _stats(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack(curves, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def _plot_roc(summary: dict, label: str) -> None:
    grid = np.linspace(0, 1, 200)
    interp = []
    plt.figure(figsize=(5, 4))
    for run in summary["runs"]:
        plt.plot(run["fpr"], run["tpr"], color="lightgray", alpha=0.4)
        interp.append(np.interp(grid, run["fpr"], run["tpr"], left=0.0, right=1.0))
    mean, std = _stats(interp)
    plt.plot(grid, mean, label=f"{label} mean")
    plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({label})")
    plt.legend()
    plt.tight_layout()


def _plot_pr(summary: dict, label: str) -> None:
    grid = np.linspace(0, 1, 200)
    interp = []
    plt.figure(figsize=(5, 4))
    for run in summary["runs"]:
        plt.plot(run["recall_curve"], run["precision_curve"], color="lightgray", alpha=0.4)
        interp.append(np.interp(grid, run["recall_curve"], run["precision_curve"], left=1.0, right=0.0))
    mean, std = _stats(interp)
    plt.plot(grid, mean, label=f"{label} mean")
    plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall ({label})")
    plt.legend()
    plt.tight_layout()


def _plot_calibration(summary: dict, label: str) -> None:
    grid = np.linspace(0, 1, 100)
    interp = []
    plt.figure(figsize=(5, 4))
    for run in summary["runs"]:
        plt.plot(run["cal_pred"], run["cal_true"], color="lightgray", alpha=0.5)
        interp.append(np.interp(grid, run["cal_pred"], run["cal_true"], left=0.0, right=1.0))
    mean, std = _stats(interp)
    plt.plot(grid, mean, label=f"{label} mean")
    plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration ({label})")
    plt.legend()
    plt.tight_layout()


def _display_summary(summary: dict, label: str) -> pd.DataFrame:
    df = _metrics_dataframe(summary)
    display(df)
    print(
        f"{label} AUC mean/std: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} | "
        f"Avg precision: {summary['ap_mean']:.4f} ± {summary['ap_std']:.4f} | "
        f"TPR@FPR mean/std: {summary['tpr_mean']:.4f} ± {summary['tpr_std']:.4f}"
    )
    return df


def _run_attack(
    mode: str,
    model_id: str,
    members_path: str | Path,
    nonmembers_path: str | Path,
    batch_cap: int = 512,
    repeats: int = 3,
    target_fpr: float = 0.01,
) -> dict:
    members_path = Path(members_path)
    nonmembers_path = Path(nonmembers_path)
    if not members_path.is_absolute():
        members_path = (ROOT / members_path).resolve()
    if not nonmembers_path.is_absolute():
        nonmembers_path = (ROOT / nonmembers_path).resolve()
    summary = evaluate_membership(
        model_id=model_id,
        members_path=members_path,
        nonmembers_path=nonmembers_path,
        batch_cap=batch_cap,
        target_fpr=target_fpr,
        repeats=repeats,
        mode=mode,
    )
    label = f"{mode.title()} attack"
    _display_summary(summary, label)
    _plot_hist(summary["runs"][0], f"Score distribution ({label})")
    _plot_roc(summary, label)
    _plot_pr(summary, label)
    _plot_calibration(summary, label)
    return summary


def run_loss_attack(model_id: str, members_path: str | Path, nonmembers_path: str | Path, batch_cap: int = 512, repeats: int = 3, target_fpr: float = 0.01) -> dict:
    """Execute loss-based membership inference and plot notebook diagnostics."""
    return _run_attack("loss", model_id, members_path, nonmembers_path, batch_cap, repeats, target_fpr)


def run_gradient_attack(model_id: str, members_path: str | Path, nonmembers_path: str | Path, batch_cap: int = 512, repeats: int = 3, target_fpr: float = 0.01) -> dict:
    """Execute gradient-based membership inference and plot notebook diagnostics."""
    return _run_attack("gradient", model_id, members_path, nonmembers_path, batch_cap, repeats, target_fpr)
