Core Concept
------------
- Goal: Attribute prompt language/model (EN vs ZH vs PRNG) from LLM-generated “random” integer sequences, including in a federated setting. Show that models/clients leak their training source via losses/gradients and even allow proportion estimation.
- Data: Synthetic sequences (len 16) from EN and ZH prompts (ARS), PRNG baselines; plus biased history variants (observational). Strong biases: EN high-value spikes (11,13,19,28,31,46,67,85,91,98), ZH low/monotonic tendencies, PRNG near-uniform.
- Attacks: Loss-based and gradient-based probes on held-out EN/ZH/PRNG sequences; mixture inference from loss features; federated-style client simulation to show the same signal in updates.

Algorithms / Implementation (what was built vs existing)
--------------------------------------------------------
- Existing: Basic sampling script for Ollama, histogram plotting utility.
- Built:
  - Integrity checks: `check_dataset.py` (shape/dtype, chi² vs uniform, z-outliers, per-position stats) with plotting helpers (`plot_global_hist`, `plot_position_means`).
  - Data splits: `make_splits.py` (90/10 train/val, external test per rep, value shift).
  - LSTM training: `train_lstm_classifier.py` (dynamic vocab shift, shared architecture; trained EN/ZH/PRNG models).
  - Probing:
    - Loss probes: `probe_losses.py` (cross-model loss matrix on held-out probes).
    - Gradient probes: `probe_gradients.py` (grad norms on probes; embedding/LSTM params).
    - Federated sim: `simulate_federated_probes.py` (multiple clients per source, local training, probe losses per client).
    - Mixture inference: `mixture_inference.py` (fit linear regressor on EN/ZH losses to recover EN proportion in mixed probes).
  - Notebook helpers: grouping plots by filename patterns for per-position means and histograms.
- Hurdles tackled:
  - Skewed/collapsed generations (ZH low/monotonic; history collapse): kept as bias observation; trained on non-history ARS sets.
  - Vocab/shift mismatches across datasets → dynamic shifting, per-model vocab alignment, clipping in probe scripts.
  - Probe attribution in federated setting → shared shift/vocab and per-client evaluation.

Suggested code/demo flow (brief):
- Show integrity check output (chi²/outliers) for EN/ZH/PRNG.
- Run/describe `train_lstm_classifier.py` on splits; note saved checkpoints.
- Show probe loss matrix (`probe_losses.py`) and gradient norms (`probe_gradients.py`).
- Show federated sim output (`simulate_federated_probes.py`).
- Show mixture inference table (`mixture_inference.py`).

Graphs / Figures to generate
----------------------------
- Per-position means (len16 main): overlay EN reps, ZH reps, PRNG reps. (Notebook group plot).
- Global histograms (len16 main): overlay EN/ZH/PRNG; optionally separate plots for history variants.
- Loss matrix heatmap: models (EN/ZH/PRNG) vs probes (EN/ZH/PRNG) from `probe_losses.py`.
- Gradient norm matrix: models vs probes from `probe_gradients.py`.
- Federated clients: table/bar showing per-client probe losses (or plot loss per probe per client source).
- Mixture inference: table of true EN proportion vs inferred proportion (from `mixture_inference.py`), optionally a line plot of true vs inferred.
- (Optional) Bias illustration: bar of top overrepresented values for EN and ZH (from integrity z-outliers).

Results and Conclusion
----------------------
- Loss-based attribution: EN model lowest on EN probes (~2.6), ZH model lowest on ZH (~2.8), PRNG high/flat (~4.6–6.4). Clear separation.
- Gradient-based attribution: smallest grad norms on own probes; much larger on others; PRNG flat with lowest norm on PRNG probes.
- Federated sim: clients trained on EN/ZH/PRNG show the same pattern—lowest loss on their source probes.
- Mixture inference: linear regression on EN/ZH losses recovers EN proportion in synthetic mixes reasonably (e.g., true 0/0.25/0.5/0.75/1 → inferred ~0.03/0.28/0.44/0.72/1.00).
- Biases: EN high-value spikes; ZH low/monotonic; history variants collapsed (note as limitations). PRNG near-uniform as a control.
- Conclusion: Prompt language leaves strong, detectable fingerprints in sequence models/clients via losses/gradients; source attribution and even mix estimation are feasible. Limitations: collapsed history prompts, strong dataset biases; future work could add cleaner history runs, per-sample MI, Monte Carlo rollouts to study prompt effects.
