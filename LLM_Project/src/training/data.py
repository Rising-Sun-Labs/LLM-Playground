import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class TextLMDataset(Dataset):
    def __init__(self, text_path, tokenizer, block_size=256):
        text = Path(text_path).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_bos_eos=False)
        self.blocks = []
        for i in range(0, max(0, len(ids)-block_size-1), block_size):
            x = ids[i:i+block_size]; y = ids[i+1:i+block_size+1]
            self.blocks.append((x,y))
        if not self.blocks:
            x = ids[:block_size] or [0]; y = ids[1:block_size+1] or [0]
            self.blocks.append((x,y))
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx):
        x,y = self.blocks[idx]
        return torch.tensor(x,dtype=torch.long), torch.tensor(y,dtype=torch.long)

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_jsonl, images_root, tokenizer, prompt="Describe the image:", img_seq_len=8, max_len=64):
        self.items = [json.loads(l) for l in Path(captions_jsonl).read_text().splitlines() if l.strip()]
        self.images_root = Path(images_root)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.img_seq_len = img_seq_len
        self.max_len = max_len
        self.tf = T.Compose([T.Resize((224,224)), T.ToTensor()])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(self.images_root / rec["image"]).convert("RGB")
        img = self.tf(img)
        inp = self.tokenizer.encode("<img> " + self.prompt, add_special_bos_eos=True, max_len=self.max_len)
        cap = self.tokenizer.encode(rec["caption"], add_special_bos_eos=True, max_len=self.max_len)
        x = (inp + cap[:-1]); y = (inp[1:] + cap)
        L = self.img_seq_len + self.max_len
        x = x[:L]; y = y[:L]
        pad_id = self.tokenizer.vocab["<pad>"]
        x += [pad_id]*(L-len(x)); y += [pad_id]*(L-len(y))
        return img, torch.tensor(x,dtype=torch.long), torch.tensor(y,dtype=torch.long)
