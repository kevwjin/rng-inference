# Prompt-Language Leakage via Membership Inference on Downstream LSTMs

This repo studies whether prompt language leaves detectable fingerprints in downstream sequence models. We sample integer sequences from an LLM in English and Chinese, train LSTMs on those sequences (and a PRNG baseline), and probe the trained models via losses and gradients. We also evaluate mixture inference and membership inference baselines.

## Environment
- Python dependencies: see `requirements.txt`. For reproducibility, use a pinned `conda` env or virtualenv.
- Llama.cpp GGUF (Ollama 3.2 3B instruct) via env vars (see `.envrc`):
  - `LLAMA_MODEL_PATH` (GGUF blob path)
  - `LLAMA_N_GPU_LAYERS`, `LLAMA_N_CTX`, `LLAMA_N_THREADS`
- Optional: use `make env` to print current env vars.

## Quick Start (Make targets)
- `make sample` — generate EN/ZH/RNG datasets (NARS, len=16, repeats=2).
- `make train` — train EN/ZH/PRNG LSTMs.
- `make rollouts` — Monte Carlo rollouts for lengths {2,4,8,16} (32 rollouts each).
- `make probe` — loss/gradient probes; saves confusion plots/JSON.
- `make figs` — regenerate rollout heatmaps and probe plots.
- `make mi` — membership inference ROC/TPR@FPR baseline.
- `make mixtures` — run mixture inference scripts.
- `make validate` — reliability check (length/range correctness) across lengths.
- `make reproduce` — chain sample → train → rollouts → probe → figs.

Defaults can be overridden, e.g. `make rollouts ROLL_SAMPLES=64` or `make probe PROBE_COUNT=0`.

## Data Generation
- Prompts: stored in `prompts/en.txt` and `prompts/zh.txt` (templates with `{seq_len}`).
- Sampling config: `configs/sampling.yaml` (NARS, range 1–100; lengths 2/4/8/16/32; T=16 chosen based on reliability/entropy; 8,192 sequences per probe set).
- Entry point: `make sample` (wraps `generate_datasets.py`). Raw/parsed NPZs saved under `artifacts/`.
- Validation: `make validate` runs `reliability_check.py` to report success/length/range errors; use this to justify rollout length cap at 16.

## Splits
- `make_splits.py` builds 90/10 train/val and a held-out probe set per source (EN/ZH/PRNG), saved under `artifacts/splits/*-split.npz`. Probe count default is full held-out; override via `--probe-count` or `PROBE_COUNT`.

## Training
- LSTM spec: 64-dim embedding → 128-unit single-layer LSTM → linear head to vocab (0–255); CE loss; Adam lr=1e-3; batch 128; grad clip 1.0; dropout 0.1.
- Train via `make train` (wraps `train_lstm_classifier.py`) for EN/ZH/PRNG checkpoints saved in `artifacts/`.

## Monte Carlo Rollouts & Entropy
- `make rollouts` runs `calc_next_token_distrib.py` for EN/ZH over lengths {2,4,8,16}, 32 rollouts each (defaults). Outputs JSONs in `rollouts/en` and `rollouts/zh`.
- Plot via `plot_rollout_heatmaps_stacked.py`; saved figures (heatmaps/entropy curves).

## Probes (Loss + Gradient)
- Loss: `probe_losses.py` (JSON) and `probe_loss_confusion.py` (PNG heatmap, `magma_r`).
- Gradient: `probe_gradients.py` + `probe_gradient_confusion.py` (PNG heatmap, `magma_r`), gradient norm over embedding + LSTM.
- Run both via `make probe` (default `PROBE_COUNT=0` uses full held-out).

## Membership Inference
- White-box baseline via `mi_eval.py` (llama.cpp logprobs). Default: members=en split, nonmembers=prng split. Override with `--members/--nonmembers`, `MI_BATCH`, `MI_FPR`.
- Outputs ROC/TPR@FPR metrics; optional histogram/ROC plots.

## Mixture Inference
- `mixture_inference.py` and `mixture_inference_gradients.py` for mixing EN/ZH probe sets and fitting linear regressors (loss and grad features). Plots calibration curves.
- Run via `make mixtures`.

## Figures & Reproduction
- Key figures:
  - Rollout heatmaps/entropy: `plot_rollout_heatmaps_stacked.py` (from `rollouts/*`).
  - Probe loss matrix: `probe_loss_confusion.py` (`artifacts/probe_losses.json`).
  - Gradient norm matrix: `probe_gradient_confusion.py`.
  - MI ROC/TPR: `mi_eval.py` (with `--plot`).
  - Calibration (mixtures): outputs from mixture scripts.
- `make figs` regenerates rollout and probe plots with current artifacts.

## Artifacts
- `artifacts/`: datasets, splits, LSTM checkpoints, probe outputs, plots.
- `rollouts/`: Monte Carlo rollout JSONs.
- `results/` (recommended): cache metrics/plots for submission; document hashes in `ARTIFACTS.md`.

## Assumptions
- White-box access to model parameters/gradients for probes/MI. Black-box extensions are out of scope here.

## License & Citation
- Add `LICENSE` and `CITATION.cff`/BibTeX as appropriate for submission.
