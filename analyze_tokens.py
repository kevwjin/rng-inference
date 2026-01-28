import os

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

single, multi = [], []
for n in range(1, 1001):
    toks = llm.tokenize(str(n).encode("utf-8"), add_bos=False)
    (single if len(toks) == 1 else multi).append((n, toks))

print('hjerr')
print(single)
print('hjerr')
print(multi)
