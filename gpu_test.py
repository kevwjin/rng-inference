from llama_cpp import Llama
llm = Llama(model_path="/usr/share/ollama/.ollama/models/blobs/sha256-9c8a9ab5edab20fcfa0e9ca322f0131c3bfb2f5a2f4ec12425a761f2f12deefa",
            n_gpu_layers=35,  # nonzero
            logits_all=True)
print("Loaded OK")
