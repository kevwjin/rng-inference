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

ARTIFACTS_ROOT = Path("artifacts")


def load_model(model_path: Path, device: torch.device) -> LstmModel:
    model = LstmModel(NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_sequence(
    model: LstmModel,
    length: int,
    device: torch.device,
    temperature: float,
) -> list[int]:
    # Random start token corresponding to class indices 0..99
    start_token = secrets.randbelow(NUM_CLASSES)
    cur_token = torch.tensor([[start_token]], device=device, dtype=torch.long)
    outputs: list[int] = [start_token + 1]

    hidden = None
    with torch.no_grad():
        for _ in range(length - 1):
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
    num_integers: int,
    seq_len: int,
    model: LstmModel,
    device: torch.device,
    temperature: float,
) -> list[list[int]]:
    return [generate_sequence(model, seq_len, device, temperature) for _ in range(num_integers)]


def resolve_model_path(prefix: str, seq_len: int, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    base_dir = ARTIFACTS_ROOT / f"{prefix}-{seq_len}"
    return base_dir / "lstm" / f"{prefix}-{seq_len}-lstm.pt"


def save_sequences(
    sequences: list[list[int]],
    prefix: str,
    seq_len: int,
    num_batches: int,
) -> Path:
    base_dir = ARTIFACTS_ROOT / f"{prefix}-{seq_len}"
    output_dir = base_dir / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{seq_len}-{num_batches}-lstm.npz"
    output_path = output_dir / filename
    np.savez(output_path, sequences=np.asarray(sequences, dtype=np.int64))
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate sequences from the trained LSTM")
    parser.add_argument(
        "--num-batches", "-b",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)",
    )
    parser.add_argument(
        "--sequence-length", "-n",
        type=int,
        default=16,
        help="Number of integers per sequence (default: 16)",
    )
    parser.add_argument(
        "--prompt-language", "-l",
        choices=("english", "en", "chinese", "zh"),
        default="english",
        help="Language tag used for naming outputs (default: english)",
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

    if args.num_batches <= 0:
        parser.error("--num-batches must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    prefix = "en" if args.prompt_language in ("english", "en") else "zh"
    model_path = resolve_model_path(prefix, args.sequence_length, args.model_path)
    if not model_path.exists():
        parser.error(f"Model weights not found: {model_path}")

    device = get_device()
    model = load_model(model_path, device)

    sequences = generate_sequences(
        args.num_batches,
        args.sequence_length,
        model,
        device,
        args.temperature,
    )
    print(json.dumps(sequences))

    if args.save_npz:
        output_path = save_sequences(
            sequences,
            prefix,
            args.sequence_length,
            args.num_batches,
        )
        print(f"Saved sequences to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
