# L1: quick test:
# text = open("input.txt", "r", encoding="utf-8").read()
#
# from L1_tokenizer import build_vocab, encode, decode
# stoi, itos = build_vocab(text)
# ids = encode("To be", stoi)
# print(ids)
# print(decode(ids, itos))

# ✅ checkpoint: encoding then decoding gives the original string.

# why char-level? zero dependencies, easiest to reason about. later you can swap this with BPE/SentencePiece.


# L2: Quick test:
from L2_data import CharDataset
ds = CharDataset("input.txt", block_size=8)
x, y = ds.get_batch("train", batch_size=4)
print(x.shape, y.shape)  # torch.Size([4, 8]) torch.Size([4, 8])
