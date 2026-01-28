PY ?= python

# Rollout defaults
ROLL_LENS ?= 2 4 8 16
ROLL_SAMPLES ?= 32
ROLL_STEPS ?= 16

# Probe defaults
PROBE_COUNT ?= 0

# MI defaults
MI_BATCH ?= 0
MI_FPR ?= 0.01

.PHONY: sample train probe rollouts figs reproduce mi mixtures env

sample:
	$(PY) generate_datasets.py --sources en zh rng --dataset-types nars --total-numbers 131072 --lengths 16 --repeats 2 --skip-existing

train:
	$(PY) train_lstm_classifier.py --data-path artifacts/splits/en-split.npz --epochs 10 --batch-size 128 --embedding-dim 64 --hidden-dim 128 --dropout 0.1
	$(PY) train_lstm_classifier.py --data-path artifacts/splits/zh-split.npz --epochs 10 --batch-size 128 --embedding-dim 64 --hidden-dim 128 --dropout 0.1
	$(PY) train_lstm_classifier.py --data-path artifacts/splits/prng-split.npz --epochs 10 --batch-size 128 --embedding-dim 64 --hidden-dim 128 --dropout 0.1

probe:
	$(PY) probe_losses.py --probe-count $(PROBE_COUNT) --out-json artifacts/probe_losses.json
	$(PY) probe_loss_confusion.py --probe-count $(PROBE_COUNT) --out-png artifacts/probe-loss-confusion.png
	$(PY) probe_gradient_confusion.py --probe-count $(PROBE_COUNT) --out-png artifacts/gradient-confusion.png

rollouts:
	@for L in $(ROLL_LENS); do \
		EN_OUT=rollouts/en/en_len$${L}_rollouts$(ROLL_SAMPLES).json; \
		ZH_OUT=rollouts/zh/zh_len$${L}_rollouts$(ROLL_SAMPLES).json; \
		$(PY) calc_next_token_distrib.py --lang en --seq-len $$L --steps $(ROLL_STEPS) --rollouts $(ROLL_SAMPLES) --output $$EN_OUT; \
		$(PY) calc_next_token_distrib.py --lang zh --seq-len $$L --steps $(ROLL_STEPS) --rollouts $(ROLL_SAMPLES) --output $$ZH_OUT; \
	done

figs:
	$(PY) plot_rollout_heatmaps_stacked.py
	$(PY) probe_loss_confusion.py --probe-count $(PROBE_COUNT) --out-png artifacts/probe-loss-confusion.png
	$(PY) probe_gradient_confusion.py --probe-count $(PROBE_COUNT) --out-png artifacts/gradient-confusion.png

mi:
	$(PY) mi_eval.py --members artifacts/splits/en-split.npz --nonmembers artifacts/splits/prng-split.npz --batch $(MI_BATCH) --fpr $(MI_FPR)

mixtures:
	$(PY) mixture_inference.py
	$(PY) mixture_inference_gradients.py

reproduce: sample train rollouts probe figs

validate:
	$(PY) reliability_check.py --runs 100 --output reliability_summary.json --resume

env:
	@echo "Python: $$($(PY) -V)"
	@echo "LLAMA_MODEL_PATH: $$LLAMA_MODEL_PATH"
	@echo "LLAMA_N_GPU_LAYERS: $$LLAMA_N_GPU_LAYERS"
	@echo "LLAMA_N_CTX: $$LLAMA_N_CTX"
	@echo "LLAMA_N_THREADS: $$LLAMA_N_THREADS"
