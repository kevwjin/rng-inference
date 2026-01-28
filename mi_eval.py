"""
Membership inference baseline using llama.cpp (forward-only).

Given two NPZ files of sequences (members vs non-members), compute an average
log-probability score per sequence under a llama.cpp model and report AUC and
TPR@FPR. Scores are higher for more likely (member) sequences.

Env vars (matching our llama.cpp setup):
  LLAMA_MODEL_PATH   path to GGUF (default: Ollama 3.2 3B instruct blob)
  LLAMA_N_GPU_LAYERS number of layers to offload (0 = CPU-only)
  LLAMA_N_CTX        context length (default 4096)
  LLAMA_N_THREADS    CPU threads (optional)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from llama_cpp import Llama

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError as e:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from e


def getenv_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


MODEL_PATH: str = os.getenv(
    "LLAMA_MODEL_PATH",
    "/usr/share/ollama/.ollama/models/blobs/sha256-9c8a9ab5edab20fcfa0e9ca322f0131c3bfb2f5a2f4ec12425a761f2f12deefa",
)
N_GPU_LAYERS: int = getenv_int("LLAMA_N_GPU_LAYERS", 0)
N_CTX: int = getenv_int("LLAMA_N_CTX", 4096)
N_THREADS: int = getenv_int("LLAMA_N_THREADS", 0)

# Init llama.cpp model; logits_all=True to capture logits at every position
llm = Llama(
    model_path=MODEL_PATH,
    logits_all=True,
    n_ctx=N_CTX,
    vocab_only=False,
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=N_THREADS or None,
)


def load_sequences(path: Path, cap: int | None = None) -> List[List[int]]:
    with np.load(path) as data:
        arr = data["sequences"]
    if cap:
        arr = arr[:cap]
    return arr.tolist()


def sequence_logprob(tokens: List[int]) -> float:
    """
    Compute total logprob of a token sequence under current llama.cpp model.
    Uses BOS by default via tokenizer; average logprob returned.
    """
    llm.reset()
    # We need logits before consuming each token; so loop token-by-token.
    logp_sum = 0.0
    # Prime BOS + tokens
    # llama_cpp tokenizer adds BOS by default; keep that behavior.
    tks = llm.tokenize(" ".join(str(x) for x in tokens).encode("utf-8"))
    # Evaluate step by step
    for i, tid in enumerate(tks):
        if i == 0:
            # First token: logits are from context=empty (or BOS if tokenizer inserted)
            llm.eval([tid])
            continue
        logits = np.array(llm.eval_logits[-1], dtype=np.float32)
        log_probs = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        logp_sum += float(log_probs[tid])
        llm.eval([tid])
    # Average logprob per token (excluding first since no logprob was added for it)
    denom = max(len(tks) - 1, 1)
    return logp_sum / denom


def compute_scores(sequences: Iterable[List[int]]) -> np.ndarray:
    scores: List[float] = []
    for seq in sequences:
        scores.append(sequence_logprob(seq))
    return np.array(scores, dtype=np.float32)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership inference via llama.cpp logprobs.")
    parser.add_argument("--members", type=Path, required=True, help="NPZ with member sequences")
    parser.add_argument("--nonmembers", type=Path, required=True, help="NPZ with non-member sequences")
    parser.add_argument("--batch", type=int, default=0, help="Optional cap on sequences per set (0 = all)")
    parser.add_argument("--fpr", type=float, default=0.01, help="Target FPR for TPR@FPR")
    parser.add_argument("--plot", type=Path, default=None, help="Optional path to save score histogram")
    args = parser.parse_args()

    members = load_sequences(args.members, cap=args.batch or None)
    nonmembers = load_sequences(args.nonmembers, cap=args.batch or None)

    y_true = np.array([1] * len(members) + [0] * len(nonmembers), dtype=np.int32)
    scores_mem = compute_scores(members)
    scores_non = compute_scores(nonmembers)
    scores = np.concatenate([scores_mem, scores_non], axis=0)

    auc = roc_auc_score(y_true, scores)
    tpr = tpr_at_fpr(y_true, scores, args.fpr)

    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR={args.fpr:.3f}: {tpr:.4f}")
    print(f"Members: n={len(members)}, mean score={scores_mem.mean():.4f}, std={scores_mem.std():.4f}")
    print(f"Non-members: n={len(nonmembers)}, mean score={scores_non.mean():.4f}, std={scores_non.std():.4f}")

    if args.plot:
        plt.figure(figsize=(6, 4))
        plt.hist(scores_mem, bins=40, alpha=0.6, label="members")
        plt.hist(scores_non, bins=40, alpha=0.6, label="non-members")
        plt.xlabel("score (avg logprob per token, BOS-included)")
        plt.ylabel("count")
        plt.title("Membership inference scores")
        plt.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        plt.close()
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
