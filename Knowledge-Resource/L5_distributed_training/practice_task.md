### Practice:
    1. DDP - ify your LM:
        - Wrap your model with DDP, and  DistributedSampler to your packed dataset, save on rank 0 only
        - Compare tokens/sec 1 GPU vs 2/4 GPUs.
    
    2. Turn on AMP + (optional) compile under DDP; record speed/memory.
    3. Switch to FSDP for a larger hidden size or sequence length that OOMs under DDP; get a successful full checkpoint.

- Stretch: multi-node run with 2 small instances; confirm gradients sync (loss curves match single-node scaled-LR behavior).

