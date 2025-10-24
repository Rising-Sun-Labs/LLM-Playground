import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model); self.proj=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        B,T,C=x.size(); qkv=self.qkv(x); q,k,v=qkv.split(C,dim=2)
        q=q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k=k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v=v.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        att=(q @ k.transpose(-2,-1))/math.sqrt(self.d_head)
        if mask is not None: att=att.masked_fill(mask==0,float('-inf'))
        att=F.softmax(att,dim=-1); att=self.drop(att)
        y=att @ v; y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self,d_model,n_heads,d_mlp,dropout=0.1):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model); self.att=CausalSelfAttention(d_model,n_heads,dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_mlp), nn.GELU(), nn.Linear(d_mlp,d_model), nn.Dropout(dropout))
    def forward(self,x,mask):
        x=x+self.att(self.ln1(x),mask); x=x+self.mlp(self.ln2(x)); return x

class MiniTransformerLM(nn.Module):
    def __init__(self,vocab_size,d_model=256,n_layers=6,n_heads=4,d_mlp=1024,max_len=256,dropout=0.1):
        super().__init__()
        self.tok=nn.Embedding(vocab_size,d_model); self.pos=nn.Embedding(max_len,d_model)
        self.blocks=nn.ModuleList([Block(d_model,n_heads,d_mlp,dropout) for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab_size,bias=False); self.max_len=max_len
    def forward(self,idx):
        B,T=idx.shape; pos=torch.arange(0,T,device=idx.device).unsqueeze(0)
        x=self.tok(idx)+self.pos(pos)
        mask=torch.tril(torch.ones(T,T,device=idx.device)).unsqueeze(0).unsqueeze(0)
        for b in self.blocks: x=b(x,mask)
        x=self.ln(x); return self.head(x)
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=50,temperature=1.0,top_k=None):
        for _ in range(max_new_tokens):
            x=idx[:,-self.max_len:]; logits=self(x)[:,-1,:]/max(temperature,1e-5)
            if top_k:
                v,_=torch.topk(logits,top_k); logits[logits<v[:,[-1]]]=-float('inf')
            probs=torch.softmax(logits,dim=-1)
            nxt=torch.multinomial(probs,1); idx=torch.cat([idx,nxt],dim=1)
        return idx
