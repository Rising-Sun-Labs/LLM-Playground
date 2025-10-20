**Tokenizer**: can mean two things in a multimodel LLM:

1. The text tokenizer(subword/bpe/unigram/byte) that turns characters -> token IDs and
2. The model token interface for images/audio (usually) just markers like <image>; sometimes discrete codes if you quantize the other modality.
