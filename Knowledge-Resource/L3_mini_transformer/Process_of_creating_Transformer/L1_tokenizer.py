#L1_tokenizer.py
def build_vocab(text: str):
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: dict[int, int]):
    return [stoi[c] for c in text]

def decode(ids: list[int], itos: dict[int, str]):
    return "".join(itos[i] for i in ids)



