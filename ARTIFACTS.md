# Artifacts and Regeneration Notes

## Checked-in
- `rollouts/en/*.json`, `rollouts/zh/*.json`: Monte Carlo rollout distributions for lengths 2/4/8/16 (32 rollouts each).
- `reliability_summary.json`: generation reliability (length/range) across lengths 2/4/6/8/16/32.
- Plots under repo root (`rollout-heatmaps-stacked.png`, etc.) and `mi-plots*` directories.
- LSTM checkpoints and splits in `artifacts/` (EN/ZH/PRNG).

## Regenerate (commands)
- Datasets: `make sample`
- Splits: `python make_splits.py` (writes `artifacts/splits/*-split.npz`)
- Training: `make train`
- Rollouts: `make rollouts`
- Probes: `make probe` (loss/grad matrices + PNGs)
- Plots: `make figs`
- Reliability: `make validate`
- Membership inference: `make mi`
- Mixture inference: `make mixtures`

## Notes
- Default probe count uses full held-out sets; override `PROBE_COUNT` to subsample.
- Rollouts assume prompts from `calc_next_token_distrib.py --lang en|zh` using internal templates (matching `sample_llm.py`).
- Environment: see `.envrc` for llama.cpp GGUF settings; run `scripts/print_env.sh` to dump versions/hashes.
