import torch
from train_lstm import LstmModel, NUM_CLASSES, get_device

def sample_sequence(
    length: int = 16,
    seed: int | None = None,
    temperature: float = 1.0,
) -> list[int]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = get_device()
    model = LstmModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("lstm_model.pt", map_location=device))
    model.eval()

    # random start token corresponding to class indices 0..99
    start_token = torch.randint(0, NUM_CLASSES, (1, 1), device=device)

    current = start_token
    outputs: list[int] = []     # start token not included

    hidden = None
    with torch.no_grad():
        for _ in range(length):
            emb = model.embedding(current)
            lstm_out, hidden = model.lstm(emb, hidden)
            logits = model.linear(lstm_out[:, -1, :])
            probs = torch.softmax(logits / temperature, dim=-1)

            idx = torch.multinomial(probs, num_samples=1)

            outputs.append(int(idx.item()) + 1)
            current = idx

    return outputs


if __name__ == "__main__":
    print(sample_sequence(length=16, seed=None))
