"""Generate integer sequences using a trained LSTM model."""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from train_lstm import LstmModel, NUM_CLASSES, get_device

ARTIFACT_ROOT = Path("artifacts")


def load_model(model_path: Path, device: torch.device) -> LstmModel:
    model = LstmModel(NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_sequence(
    model: LstmModel,
    seq_len: int,
    device: torch.device,
    temperature: float,
) -> list[int]:
    # Random start token corresponding to class indices 0..99
    start_token = secrets.randbelow(NUM_CLASSES)
    cur_token = torch.tensor([[start_token]], device=device, dtype=torch.long)
    outputs: list[int] = [start_token + 1]

    hidden = None
    with torch.no_grad():
        for _ in range(seq_len - 1):
            emb = model.embedding(cur_token)
            lstm_out, hidden = model.lstm(emb, hidden)
            logits = model.linear(lstm_out[:, -1, :])
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt_token = torch.multinomial(probs, num_samples=1)
            token = int(nxt_token.item())
            outputs.append(token + 1)
            cur_token = nxt_token

    return outputs


def generate_sequences(
    num_seqs: int,
    seq_len: int,
    model: LstmModel,
    device: torch.device,
    temperature: float,
) -> list[list[int]]:
    return [generate_sequence(model, seq_len, device, temperature) for _ in range(num_seqs)]


def resolve_model_path(
    prefix: str,
    seq_len: int,
    model_path: Path | None
) -> Path:
    if model_path is not None:
        return model_path
    base_dir = ARTIFACT_ROOT / f"{prefix}-{seq_len}"
    return base_dir / "lstm" / f"{prefix}-{seq_len}-lstm.pt"


def save_sequences(
    seqs: list[list[int]],
    prefix: str,
    seq_len: int,
    num_seqs: int,
) -> Path:
    base_dir = ARTIFACT_ROOT / f"{prefix}-{seq_len}"
    output_dir = base_dir / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{seq_len}-lstm-{num_seqs}.npz"
    output_path = output_dir / filename
    seqs_np = np.asarray(seqs, dtype=np.int64)
    np.savez(output_path, sequences=seqs_np)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate sequences from the trained LSTM")
    parser.add_argument(
        "--prompt-language", "-l",
        choices=("english", "en", "chinese", "zh"),
        default="english",
        help="Language tag used for naming outputs (default: english)",
    )
    parser.add_argument(
        "--sequence-length", "-n",
        type=int,
        default=16,
        help="Number of integers per sequence (default: 16)",
    )
    parser.add_argument(
        "--num-sequences", "-s",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save sequences under artifacts/<lang>-<length>-lstm/",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional explicit path to trained LSTM weights",
    )
    args = parser.parse_args(argv)

    prefix = "en" if args.prompt_language in ("english", "en") else "zh"
    model_path = resolve_model_path(
        prefix, args.sequence_length, args.model_path)
    if not model_path.exists():
        parser.error(f"Model weights not found: {model_path}")

    device = get_device()
    model = load_model(model_path, device)

    seqs = generate_sequences(
        args.num_sequences,
        args.sequence_length,
        model,
        device,
        args.temperature,
    )
    print(json.dumps(seqs))

    if args.save_npz:
        output_path = save_sequences(
            seqs,
            prefix,
            args.sequence_length,
            args.num_sequences
        )
        print(f"Saved sequences to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
