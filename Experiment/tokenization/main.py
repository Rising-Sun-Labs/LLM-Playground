from bpe_tokenizer import BPETokenizer

#  Train
tok = BPETokenizer()
tok.train(corpus=open("my_corpus.txt", "r", encoding="utf-8"),
          vocab_size=64000,
          special_tokens=["<bos>","<eos>","<pad>","<usr>","<asst>"])
tok.save("my_tokenizer.json")

# Load later
tok = BPETokenizer.load("my_tokenizer.json")
ids = tok.encode("<usr>please review this diff</usr>")
text = tok.decode(ids)


# Design notes & extensions

# Lossless: because we operate at bytes + </w>, anything encodable in UTF-8 round-trips.

# Whitespace: we keep whitespace chunks as raw bytes, so spacing/newlines survive exactly.

# Normalization: you can turn normalize=False to avoid NFKC; for code-heavy corpora, many teams prefer raw.

# Speed: this is educational-quality Python. For production, you’ll want:

# a faster pair counter (NumPy/Numba/Rust),

# multi-threaded training,

# memory-mapped corpora and batched stats.

# Vocabulary growth: BPE merges stop when you hit your target (or run out of pairs).

# Special tokens: here we treat specials as literal spans in text. In production, you usually tokenize a structured chat format and inject specials by the formatter rather than hoping to see literal strings.