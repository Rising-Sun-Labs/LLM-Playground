# In-House Multimodal LLM (Blueprint)

Text → BPE → IDs → Decoder-only Transformer (causal).  
Image → Tiny CNN → feature vectors → **prefix** added to first k token positions.

**Training plan**
1) Train tokenizer on your corpus
2) Pretrain LM (text only)
3) Fine-tune captioning (image+text)
4) Task finetunes (code review, emails)

**Serving**
- Local FastAPI (offline)
- Checkpoints on disk
- DDP for multi-GPU
- Monitoring with TensorBoard
