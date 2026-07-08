"""Train an LM to map (policy code, params) -> (execution trace, updated params).

HF backend (default): fine-tune a pretrained Llama-family (or similar) causal LM:
    python3 -m ntp.train --data data/demo --out runs/smol135 --steps 1500 --batch-size 4

Variants:
    --hf-model HuggingFaceTB/SmolLM2-360M [--lora]
    --hf-model Qwen/Qwen2.5-0.5B --lora --grad-ckpt
    --hf-model meta-llama/Llama-3.2-1B --lora --grad-ckpt   (gated: needs HF_TOKEN)

Scratch backend (offline fallback; from-scratch char-level transformer):
    python3 -m ntp.train --backend scratch --data data/demo --out runs/scratch \
        --steps 3000 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from . import wb
from .scratch_model import (CharTokenizer, MiniGPT, ModelConfig, pick_device,
                            save_checkpoint)


def load_examples(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_train_examples(args) -> List[dict]:
    ex = load_examples(os.path.join(args.data, "train.jsonl"))
    for p in (getattr(args, "extra_train", None) or []):
        extra = load_examples(p)
        print("extra train: %s (+%d examples)" % (p, len(extra)), flush=True)
        ex.extend(extra)
    return ex


# ---------------------------------------------------------------------------
# Scratch backend
# ---------------------------------------------------------------------------

def tokenize_examples(examples: List[dict], tok: CharTokenizer) -> List[Tuple[List[int], int]]:
    """Returns (full_ids, prompt_len) per example."""
    data = []
    for ex in examples:
        p = tok.encode(ex["prompt"])
        t = tok.encode(ex["target"])
        data.append((p + t, len(p)))
    return data


def collate(batch, pad: int):
    """Items are (ids, prompt_len, ...) — extra fields (e.g. weight offsets) ignored."""
    maxlen = max(len(item[0]) for item in batch)
    x = torch.full((len(batch), maxlen - 1), pad, dtype=torch.long)
    y = torch.full((len(batch), maxlen - 1), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        ids, plen = item[0], item[1]
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        yy = torch.tensor(ids[1:], dtype=torch.long)
        yy[:plen - 1] = -100  # no loss on prompt tokens
        y[i, :n - 1] = yy
    return x, y


def make_batches(data: List[Tuple[List[int], int]], batch_size: int, rng: random.Random):
    """Length-bucketed batches, shuffled order (cheap padding reduction)."""
    idx = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate_val(model, val_data, batch_size, pad, device, limit=200):
    model.eval()
    total_loss, total_tok, total_correct = 0.0, 0, 0
    data = val_data[:limit]
    for i in range(0, len(data), batch_size):
        x, y = collate(data[i:i + batch_size], pad)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-100, reduction="sum")
        mask = y != -100
        total_loss += loss.item()
        total_tok += int(mask.sum().item())
        total_correct += int(((logits.argmax(-1) == y) & mask).sum().item())
    model.train()
    return total_loss / max(1, total_tok), total_correct / max(1, total_tok)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def train_scratch(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_ex = load_train_examples(args)
    val_ex = load_examples(os.path.join(args.data, "val.jsonl"))
    tok = CharTokenizer()
    train_data = tokenize_examples(train_ex, tok)
    val_data = tokenize_examples(val_ex, tok)

    max_len = max(len(ids) for ids, _ in train_data + val_data)
    ctx = min(args.ctx, ((max_len + 63) // 64) * 64)
    n0 = len(train_data) + len(val_data)
    train_data = [d for d in train_data if len(d[0]) <= ctx]
    val_data = [d for d in val_data if len(d[0]) <= ctx]
    dropped = n0 - len(train_data) - len(val_data)
    if dropped:
        print("WARNING: dropped %d examples longer than ctx=%d (raise --ctx)"
              % (dropped, ctx), flush=True)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                      n_layer=args.n_layer, n_head=args.n_head,
                      max_len=ctx, dropout=args.dropout)
    model = MiniGPT(cfg).to(device)
    print("device=%s params=%.2fM ctx=%d train_ex=%d val_ex=%d max_seq=%d"
          % (device, model.num_params() / 1e6, ctx, len(train_data), len(val_data), max_len))

    run = wb.init_run("train", wb.tag_from_out(args.out), dict(
        vars(args), backend="scratch", device=device, n_params=model.num_params(),
        ctx=ctx, train_examples=len(train_data), val_examples=len(val_data),
        **wb.data_meta(args.data)), enabled=not args.no_wandb)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a")

    model.train()
    step = 0
    best_val = float("inf")
    batches = make_batches(train_data, args.batch_size, rng)
    bi = 0
    t0 = time.time()
    tok_count = 0
    loss_acc, loss_n = 0.0, 0

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if bi >= len(batches):
                batches = make_batches(train_data, args.batch_size, rng)
                bi = 0
            batch = [train_data[i] for i in batches[bi]]
            bi += 1
            x, y = collate(batch, tok.PAD)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   ignore_index=-100)
            (loss / args.accum).backward()
            tok_count += x.numel()
            loss_acc += loss.item()
            loss_n += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            msg = {"step": step, "loss": round(loss_acc / max(1, loss_n), 4),
                   "lr": round(lr, 6), "tok_per_s": int(tok_count / dt),
                   "eta_min": round((args.steps - step) * dt / step / 60, 1)}
            print(json.dumps(msg), flush=True)
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            wb.log(run, {"train/loss": loss_acc / max(1, loss_n), "train/lr": lr,
                         "train/grad_norm": float(grad_norm),
                         "train/tok_per_s": tok_count / dt,
                         "train/epoch": step * args.batch_size * args.accum / len(train_data),
                         "train/eta_min": msg["eta_min"]}, step=step)
            loss_acc, loss_n = 0.0, 0

        if step % args.val_every == 0 or step == args.steps:
            vl, va = evaluate_val(model, val_data, args.batch_size, tok.PAD, device)
            msg = {"step": step, "val_loss": round(vl, 4), "val_tok_acc": round(va, 4)}
            print(json.dumps(msg), flush=True)
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            if vl < best_val:
                best_val = vl
                save_checkpoint(os.path.join(args.out, "best.pt"), model, step,
                                {"val_loss": vl, "val_tok_acc": va, "backend": "scratch"})
            wb.log(run, {"val/loss": vl, "val/tok_acc": va, "val/best_loss": best_val},
                   step=step)
    save_checkpoint(os.path.join(args.out, "last.pt"), model, step,
                    {"backend": "scratch"})
    print("done. best val loss %.4f. checkpoints in %s" % (best_val, args.out))
    wb.set_summary(run, {"best_val_loss": best_val})
    wb.finish(run)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "scratch"], default="hf")
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra-train", nargs="*", default=None,
                    help="additional train jsonl files (e.g. DAgger corrections)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging (project: $WANDB_PROJECT, "
                         "default 'neural-trace-policies')")
    # scratch model size
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--ctx", type=int, default=3328)
    # hf backend
    ap.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M",
                    help="any HF causal LM id (Llama-family recommended)")
    ap.add_argument("--hf-lr", type=float, default=1e-4)
    ap.add_argument("--lora", action="store_true",
                    help="LoRA fine-tuning (recommended for models >=360M on a laptop)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (trade speed for memory)")
    ap.add_argument("--delta-token-weight", type=float, default=1.0,
                    help="HF backend: loss weight for the <DELTA>/<PARAMS> block tokens "
                         "(magnitude digits); trace tokens keep weight 1.0")
    args = ap.parse_args()

    if args.backend == "scratch":
        train_scratch(args)
    else:
        from .hf_backend import train_hf
        train_hf(args)


if __name__ == "__main__":
    main()
