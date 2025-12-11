"""
Next-token distribution + Monte Carlo rollouts using llama.cpp (GGUF).

Env vars:
  LLAMA_MODEL_PATH   path to GGUF (default: Ollama 3.2 3B instruct blob)
  LLAMA_N_GPU_LAYERS number of layers to offload (0 = CPU-only)
  LLAMA_N_CTX        context length (default 4096)
  LLAMA_N_THREADS    CPU threads (optional)

WARNING: Multi-token per-integer probabilities are extremely expensive because
we clone/restore full KV state per integer per step.
"""

import json
import math
import os
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
from llama_cpp import Llama


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
    "/usr/share/ollama/.ollama/models/blobs/sha256-"
    "9c8a9ab5edab20fcfa0e9ca322f0131c3bfb2f5a2f4ec12425a761f2f12deefa",
)
N_GPU_LAYERS: int = getenv_int("LLAMA_N_GPU_LAYERS", 26)
N_CTX: int = getenv_int("LLAMA_N_CTX", 4096)
N_THREADS: int = getenv_int("LLAMA_N_THREADS", 0)

# init llama.cpp model; logits_all=True to capture logits at every position
llm = Llama(
    model_path=MODEL_PATH,
    logits_all=True,
    n_ctx=N_CTX,
    vocab_only=False,
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=N_THREADS,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / exps.sum()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    m = logits.max()
    return logits - m - math.log(np.exp(logits - m).sum())


def get_next_token_distribution(prompt: str):
    """Return (probs, logits, next_token_id) for the next token after prompt."""
    llm.reset()
    tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.eval(tokens)
    logits = np.array(llm.eval_logits[-1], dtype=np.float32)
    probs = _softmax(logits)
    next_token_id = int(probs.argmax())
    return probs, logits, next_token_id


def _integer_to_token_ids(nums: Iterable[int]) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    for n in nums:
        toks = llm.tokenize(str(n).encode("utf-8"))
        if len(toks) == 0:
            continue
        mapping[n] = toks
    return mapping


def _prob_of_token_sequence(token_ids: Sequence[int], state) -> float:
    """
    Compute probability of a token sequence given cached state.
    WARNING: uses save_state/load_state -> very slow.
    """
    llm.load_state(state)
    logp = 0.0
    for tid in token_ids:
        logits = np.array(llm.eval_logits[-1], dtype=np.float32)
        logp += _log_softmax(logits)[tid]
        llm.eval([tid])
    return math.exp(logp)


def monte_carlo_rollouts(
    prompt: str,
    steps: int,
    n_samples: int,
    tracked_integers: Iterable[int] = range(1, 101),
    *,
    checkpoint_path: str | None = None,
    resume_sums: List[Dict[int, float]] | None = None,
    resume_sequences: List[List[int]] | None = None,
):
    """
    Monte Carlo rollouts with per-integer probabilities (multi-token aware).

    This clones/restores full state per integer per step: expect slow runtimes.
    """
    int_token_map = _integer_to_token_ids(tracked_integers)

    llm.reset()
    tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.eval(tokens)
    base_state = llm.save_state()

    per_step_prob_sums: List[Dict[int, float]] = (
        resume_sums
        if resume_sums is not None
        else [dict.fromkeys(int_token_map.keys(), 0.0) for _ in range(steps)]
    )
    sampled_sequences: List[List[int]] = resume_sequences[:] if resume_sequences else []
    start_rollout = len(sampled_sequences)

    for rollout_idx in range(start_rollout, n_samples):
        llm.load_state(base_state)
        seq: List[int] = []

        for step_idx in range(steps):
            logits = np.array(llm.eval_logits[-1], dtype=np.float32)
            probs = _softmax(logits)

            step_state = llm.save_state()
            for n, toks in int_token_map.items():
                prob = _prob_of_token_sequence(toks, step_state)
                per_step_prob_sums[step_idx][n] += prob
            llm.load_state(step_state)

            next_token_id = int(np.random.choice(len(probs), p=probs))
            seq.append(next_token_id)
            llm.eval([next_token_id])

        sampled_sequences.append(seq)

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "prompt": prompt,
                        "steps": steps,
                        "rollouts": n_samples,
                        "min_int": min(tracked_integers),
                        "max_int": max(tracked_integers),
                        "completed": rollout_idx + 1,
                        "per_step_prob_sums": per_step_prob_sums,
                        "sampled_sequences": sampled_sequences,
                    },
                    f,
                )

    per_step_probs: List[Dict[int, float]] = []
    completed = len(sampled_sequences)
    if completed == 0:
        return per_step_probs, sampled_sequences

    for step_idx in range(steps):
        avg = {
            n: prob_sum / float(completed)
            for n, prob_sum in per_step_prob_sums[step_idx].items()
        }
        per_step_probs.append(avg)

    llm.load_state(base_state)
    return per_step_probs, sampled_sequences


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Monte Carlo next-token distributions via llama.cpp GGUF.")
    parser.add_argument("--prompt", help="Prompt text to condition on")
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "english", "chinese"],
        help="If --prompt is not set, build a prompt using sample_llm builders",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Sequence length to build prompt with (when --prompt is omitted)",
    )
    parser.add_argument("--steps", type=int, required=True, help="Number of rollout steps")
    parser.add_argument("--rollouts", type=int, default=1, help="Number of rollouts (n_samples)")
    parser.add_argument("--min-int", type=int, default=1, help="Start of tracked integer range (inclusive)")
    parser.add_argument("--max-int", type=int, default=100, help="End of tracked integer range (inclusive)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for long runs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        if not args.lang or not args.seq_len:
            parser.error("Provide --prompt or (--lang and --seq-len)")
        from sample_llm import build_prompt_en, build_prompt_zh

        lang = args.lang.lower()
        if lang in ("en", "english"):
            prompt = build_prompt_en(args.seq_len)
        else:
            prompt = build_prompt_zh(args.seq_len)

    tracked = range(args.min_int, args.max_int + 1)

    resume_sums = None
    resume_sequences = None
    completed = 0
    ckpt_path = args.checkpoint
    if ckpt_path is None and args.output:
        ckpt_path = args.output + ".checkpoint.json"

    if args.resume and ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            data = json.load(f)
        # basic consistency checks
        if data.get("prompt") != prompt or data.get("steps") != args.steps:
            raise SystemExit("Checkpoint mismatch: prompt or steps differ.")
        if data.get("min_int") != args.min_int or data.get("max_int") != args.max_int:
            raise SystemExit("Checkpoint mismatch: tracked integer range differs.")
        resume_sums = data.get("per_step_prob_sums")
        resume_sequences = data.get("sampled_sequences")
        completed = int(data.get("completed", 0))
        print(f"Resuming from checkpoint {ckpt_path} with {completed}/{args.rollouts} rollouts", file=sys.stderr)

    per_step_probs, sampled = monte_carlo_rollouts(
        prompt=prompt,
        steps=args.steps,
        n_samples=args.rollouts,
        tracked_integers=tracked,
        checkpoint_path=ckpt_path,
        resume_sums=resume_sums,
        resume_sequences=resume_sequences,
    )

    result = {
        "prompt": prompt,
        "steps": args.steps,
        "rollouts": args.rollouts,
        "min_int": args.min_int,
        "max_int": args.max_int,
        "per_step_probs": per_step_probs,
        "sampled_sequences": sampled,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result))
