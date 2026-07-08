"""Primary backend: fine-tune a pretrained (Llama-family or similar) causal LM.

Tested variants (any HF causal LM id works via --hf-model):
    HuggingFaceTB/SmolLM2-135M   Llama arch, full fine-tune, default
    HuggingFaceTB/SmolLM2-360M   Llama arch, full FT or --lora
    Qwen/Qwen2.5-0.5B            --lora --grad-ckpt recommended on laptop GPUs
    meta-llama/Llama-3.2-1B      gated (accept license + HF_TOKEN), --lora --grad-ckpt

LoRA checkpoints are saved as adapters; inference merges them automatically.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from . import wb

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _device(pref=None):
    from .scratch_model import pick_device
    return pick_device(pref)


def _encode_prompt(tok, prompt: str) -> List[int]:
    ids = tok(prompt, add_special_tokens=False).input_ids
    if tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + ids
    return ids


def hf_tokenize(examples: List[dict], tok) -> List[Tuple[List[int], int, int]]:
    """Returns (full_ids, prompt_len, weight_start) per example.

    The target is tokenized in two parts split at its <DELTA>/<PARAMS> block so the
    parameter-update tokens form their own region; weight_start is the absolute index
    of that region (used by --delta-token-weight to upweight magnitude digits).
    """
    data = []
    for ex in examples:
        p = _encode_prompt(tok, ex["prompt"])
        tgt = ex["target"]
        mark = "<DELTA>" if "<DELTA>" in tgt else ("<PARAMS>" if "<PARAMS>" in tgt else None)
        if mark:
            i = tgt.index(mark)
            t1 = tok(tgt[:i], add_special_tokens=False).input_ids
            t2 = tok(tgt[i:], add_special_tokens=False).input_ids + [tok.eos_token_id]
            data.append((p + t1 + t2, len(p), len(p) + len(t1)))
        else:
            t = tok(tgt, add_special_tokens=False).input_ids + [tok.eos_token_id]
            data.append((p + t, len(p), len(p) + len(t)))
    return data


def load_hf(model_id: str, device: str, lora: bool, grad_ckpt: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora:
            model.enable_input_require_grads()
    return tok, model.to(device)


def train_hf(args):
    from .train import collate, load_examples, load_train_examples, lr_at, make_batches

    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok, model = load_hf(args.hf_model, device, args.lora, args.grad_ckpt)
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_data = hf_tokenize(load_train_examples(args), tok)
    val_data = hf_tokenize(load_examples(os.path.join(args.data, "val.jsonl")), tok)
    print("device=%s model=%s params=%.0fM trainable=%.1fM train_ex=%d max_tok=%d"
          % (device, args.hf_model, n_param / 1e6, n_train / 1e6, len(train_data),
             max(len(i) for i, _ in train_data)), flush=True)

    run = wb.init_run("train", wb.tag_from_out(args.out), dict(
        vars(args), backend="hf", device=device, n_params=n_param,
        n_trainable=n_train, train_examples=len(train_data),
        val_examples=len(val_data),
        max_tokens=max(len(i) for i, _ in train_data),
        **wb.data_meta(args.data)), enabled=not args.no_wandb)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.hf_lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a")
    pad = tok.pad_token_id
    out_dir = os.path.join(args.out, "hf_model")

    model.train()
    batches = make_batches(train_data, args.batch_size, rng)
    bi = 0
    best_val = float("inf")
    t0 = time.time()
    loss_acc, loss_n = 0.0, 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if bi >= len(batches):
                batches = make_batches(train_data, args.batch_size, rng)
                bi = 0
            batch = [train_data[i] for i in batches[bi]]
            bi += 1
            x, y = collate(batch, pad)
            x, y = x.to(device), y.to(device)
            logits = model(input_ids=x, attention_mask=(x != pad).long()).logits
            dtw = getattr(args, "delta_token_weight", 1.0)
            if dtw != 1.0:
                w = torch.ones_like(y, dtype=torch.float)
                for i, item in enumerate(batch):
                    if len(item) > 2:  # weight region begins at label index ws-1
                        w[i, max(0, item[2] - 1):] = dtw
                w[y == -100] = 0.0
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                     ignore_index=-100, reduction="none").view_as(w)
                loss = (ce * w).sum() / w.sum().clamp(min=1.0)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                       ignore_index=-100)
            (loss / args.accum).backward()
            loss_acc += loss.item()
            loss_n += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        lr = lr_at(step - 1, args.hf_lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            msg = {"step": step, "loss": round(loss_acc / max(1, loss_n), 4),
                   "lr": round(lr, 7),
                   "sec_per_step": round(dt / step, 2),
                   "eta_min": round((args.steps - step) * dt / step / 60, 1)}
            print(json.dumps(msg), flush=True)
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            wb.log(run, {"train/loss": loss_acc / max(1, loss_n), "train/lr": lr,
                         "train/grad_norm": float(grad_norm),
                         "train/sec_per_step": dt / step,
                         "train/epoch": step * args.batch_size * args.accum / len(train_data),
                         "train/eta_min": msg["eta_min"]}, step=step)
            loss_acc, loss_n = 0.0, 0

        if step % args.val_every == 0 or step == args.steps:
            vl, va = _eval_val(model, val_data, args.batch_size, pad, device)
            msg = {"step": step, "val_loss": round(vl, 4), "val_tok_acc": round(va, 4)}
            print(json.dumps(msg), flush=True)
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            if vl < best_val:
                best_val = vl
                model.save_pretrained(out_dir)  # adapter only when LoRA
                tok.save_pretrained(out_dir)
                with open(os.path.join(out_dir, "ntp_meta.json"), "w") as f:
                    json.dump({"backend": "hf", "hf_model": args.hf_model,
                               "lora": args.lora, "step": step, "val_loss": vl}, f)
            wb.log(run, {"val/loss": vl, "val/tok_acc": va, "val/best_loss": best_val},
                   step=step)
    print("done. best val loss %.4f -> %s" % (best_val, out_dir), flush=True)
    wb.set_summary(run, {"best_val_loss": best_val})
    wb.finish(run)


@torch.no_grad()
def _eval_val(model, val_data, batch_size, pad, device, limit=128):
    from .train import collate
    model.eval()
    tot, ntok, ncor = 0.0, 0, 0
    for i in range(0, min(len(val_data), limit), batch_size):
        x, y = collate(val_data[i:i + batch_size], pad)
        x, y = x.to(device), y.to(device)
        logits = model(input_ids=x, attention_mask=(x != pad).long()).logits
        l = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                            ignore_index=-100, reduction="sum")
        mask = y != -100
        tot += l.item()
        ntok += int(mask.sum())
        ncor += int(((logits.argmax(-1) == y) & mask).sum())
    model.train()
    return tot / max(1, ntok), ncor / max(1, ntok)


class HFGenerator:
    """Greedy generation wrapper; merges LoRA adapters automatically."""

    def __init__(self, model_dir: str, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(model_dir)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.config.use_cache = True
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int, stop: str = "<END>") -> str:
        return self.generate_batch([prompt], max_new, stop)[0]

    @torch.no_grad()
    def generate_batch(self, prompts, max_new: int, stop: str = "<END>"):
        """Greedy decode a batch of prompts (left-padded)."""
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None \
            else self.tok.eos_token_id
        enc = [_encode_prompt(self.tok, p) for p in prompts]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, width - len(e):] = torch.tensor(e, dtype=torch.long)
            mask[i, width - len(e):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        out = self.model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=pad, eos_token_id=self.tok.eos_token_id)
        gens = []
        for i in range(len(enc)):
            gen = self.tok.decode(out[i][width:], skip_special_tokens=True)
            if stop in gen:
                gen = gen.split(stop, 1)[0] + stop
            gens.append(gen)
        return gens
