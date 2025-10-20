# How to Train a Tokenizer (Beginner → Advanced)

Transformers work on integer IDs; tokenization maps text → subword IDs. We’ll implement **BPE** (Byte Pair Encoding) from scratch for **text** and **code**.

## Why BPE?
- Reduces vocabulary while keeping expressiveness
- Handles rare words via subwords
- Works well for **code** (symbols stay intact)

## Special tokens
- <pad> (padding), <bos> (begin), <eos> (end), <unk> (unknown), <img> (image marker)

## BPE training (concept)
1. Start with characters + `</w>` (end-of-word).
2. Count adjacent pairs over the corpus.
3. Merge most frequent pair → new symbol.
4. Repeat until vocab_size or min_freq.
5. Save `vocab` + `merges`.

## Encoding
Apply merges greedily to each word; add <bos>/<eos> if needed.

## Decoding
Map ids → symbols, join until `</w>` per word.

## Practical pipeline
- Gather raw text: docs, emails, code files.
- Clean lightly (normalize whitespace), **don’t remove punctuation** for code.
- Train BPE on your text.
- Inspect vocab and merges; ensure code symbols appear.

## Exercises
1) Train 4k/8k/32k vocabularies on the same corpus; compare LM perplexity.
2) Train a **code-specialized** tokenizer using only code files; compare encode length on code snippets.
