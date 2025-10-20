1. Granularity Families
   1.1. Word-level (space/punct rules; e.g., Moses, spaCy)
   1.2. Character level
   1.3. Byte-level

2. Subword algorithms(LLM workhorses)
   2.1. BPE (Byte-Pair Encoding)
   2.2. Unigram LM (SentencePiece)
   2.3. WordPiece
   2.4. SentencePiece Framework


| Goal                                 | Best bet                                                     | Why / Notes                                                             |
| ------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| General multilingual + code          | **Unigram (SentencePiece)** with **byte-fallback**, 64k–100k | Great compression, robust; can sample for regularization                |
| English-heavy, OpenAI-compat prompts | **Byte-Level BPE**                                           | Matches tiktoken style; simple + fast                                   |
| Strict robustness to any bytes/logs  | **Byte-Level BPE** (no normalization)                        | Lossless & resilient                                                    |
| Code-centric (lots of identifiers)   | **Unigram** or **BPE** + lexer pre-split                     | Keep operators/keywords compact; subword long identifiers               |
| Data augmentation during encode      | **BPE-dropout** or **Unigram sampling**                      | Adds variability; helps generalization                                  |
| Multimodal images/audio              | **Markers + encoders**; optionally **VQ** later              | Start simple; add discrete codes if you really need token serialization |
