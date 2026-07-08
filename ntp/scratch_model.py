"""From-scratch char-level decoder-only transformer (no downloads, runs on MPS/CPU/CUDA).

Small on purpose: the corpus is a narrow synthetic format, so a ~6M-param model learns
the structure quickly and the whole pipeline stays self-contained.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """Fixed vocab: PAD + newline + printable ASCII 32..126."""

    PAD = 0

    def __init__(self):
        self.chars = ["\x00", "\n"] + [chr(c) for c in range(32, 127)]
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids if i != self.PAD)


@dataclass
class ModelConfig:
    vocab_size: int = 97
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    max_len: int = 3072
    dropout: float = 0.05


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nh = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # prefill / training: q covers the same positions as k -> causal mask.
        # single-token decode with cache: attend to the full past, no mask.
        # no in-attention dropout: keeps MPS on the memory-efficient SDPA kernel
        causal = q.size(2) == k.size(2) and q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, pos_start: int = 0,
                caches: Optional[List[dict]] = None) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(pos_start, pos_start + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache=None if caches is None else caches[i])
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids: List[int], tokenizer: CharTokenizer, max_new: int,
                 stop: str = "<END>", device: str = "cpu") -> str:
        """Greedy decode with a per-layer KV cache; stops on `stop` or length limit."""
        self.eval()
        caches: List[dict] = [{} for _ in self.blocks]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = self.forward(x, pos_start=0, caches=caches)
        out_ids: List[int] = []
        text = ""
        cur = int(logits[0, -1].argmax().item())
        pos = len(ids)
        for _ in range(max_new):
            if pos >= self.cfg.max_len:
                break
            out_ids.append(cur)
            text += tokenizer.chars[cur]
            if text.endswith(stop):
                break
            x = torch.tensor([[cur]], dtype=torch.long, device=device)
            logits = self.forward(x, pos_start=pos, caches=caches)
            cur = int(logits[0, -1].argmax().item())
            pos += 1
        return text


def save_checkpoint(path: str, model: MiniGPT, step: int, extra: Optional[dict] = None):
    torch.save({
        "model_config": asdict(model.cfg),
        "model_state": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def pick_device(pref: Optional[str] = None) -> str:
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
