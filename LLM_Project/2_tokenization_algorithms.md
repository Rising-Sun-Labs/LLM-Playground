# Tokenization Algorithms (Text & Images)

## Text
| Algorithm  | How it trains                  | Speed | Quality | Notes |
|------------|--------------------------------|-------|---------|-------|
| BPE        | Greedy pair merges             | Fast  | Good    | Simple, great for code & NL |
| WordPiece  | Merges to maximize likelihood  | Fast  | Good    | BERT family |
| Unigram LM | Probabilistic inventory pruning| Med   | Great   | Strong quality, more complex |

**Choose BPE** first for a from-scratch project; expand to Unigram later if needed.

## Images
- **Continuous**: Image encoder (CNN/ViT) → vectors; fuse into LM (image-prefix). *We implement this.*
- **Discrete**: VQ tokenizers (visual tokens) for image generation; more complex.
