# train_fsdp.py

# When to use:
# 1. Model doesn't fit on one GPU (even with small batch)
# 2. You need to raise sequence length or hidden size and DDP OOMs.
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
import torchvision as tv

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class BigNet(nn.Module):
    def __init__(self, width=4096):
        super().__init__()
        self.embed = nn.Embedding(5000, width)
        self.blocks = nn.Sequential(
            nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width),
            nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width),
        )
        self.head = nn.Linear(width, 10)
    def forward(self, x):
        # fake token input; for vision/text replace with real model
        h = self.embed(x)          # [B,T,D]
        h = h.mean(dim=1)          # pretend pooling
        h = self.blocks(h)
        return self.head(h)

def main():
    local_rank = setup()
    device = f"cuda:{local_rank}"

    # Data
    tf = tv.transforms.ToTensor()
    train = tv.datasets.MNIST("./data", train=True, download=True, transform=tf)
    sampler = DistributedSampler(train, shuffle=True, drop_last=True)
    dl = DataLoader(train, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    # Model
    model = BigNet(width=4096).to(device)

    # Auto-wrap big submodules (you’d typically wrap Transformer blocks)
    policy = size_based_auto_wrap_policy(min_num_params=1_000_000)

    # Mixed precision inside FSDP
    mp = MixedPrecision(param_dtype=torch.bfloat16,  # parameters
                        reduce_dtype=torch.bfloat16, # gradient comm
                        buffer_dtype=torch.bfloat16) # buffers

    model = FSDP(model, auto_wrap_policy=policy, mixed_precision=mp)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        model.train()
        for xb, yb in dl:
            # Here, xb are images by default; for demo, make fake token ids
            B = xb.size(0)
            fake_tokens = torch.randint(0, 5000, (B, 128), device=device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(fake_tokens)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
        if local_rank == 0:
            print("epoch", epoch, "done")

    # Checkpointing — FSDP: save a full state dict from rank 0 (gathers shards)
    if local_rank == 0:
        full_state = model.state_dict()  # by default, “full” (gathers) in recent PyTorch
        torch.save({"model": full_state}, "bignet_fsdp_full.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
