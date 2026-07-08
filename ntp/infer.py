"""Iterative inference: the trained LM *is* the trainer.

For each eval task, start from the task's initial parameters and repeatedly feed
(code, current params) to the model; it emits (predicted trace, updated params), and
the updated params are fed back in. After N rounds the final params are a
model-trained policy — verify them with ntp.evaluate / ntp.run_policy.

Usage:
    python3 -m ntp.infer --ckpt runs/demo/best.pt --tasks data/demo/eval_tasks.json \
        --out runs/demo/rollouts.json --limit 25
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .textio import (build_prompt, build_target, params_to_text, parse_output,
                     parse_params_text, shapes_for)


def make_generator(ckpt_path: str, device_pref=None):
    """Returns (generate(prompt, max_new) -> str, device, backend_name)."""
    if os.path.isdir(ckpt_path):
        from .hf_backend import HFGenerator
        gen = HFGenerator(ckpt_path, device=device_pref)
        return gen.generate, gen.device, "hf"
    from .scratch_model import CharTokenizer, load_checkpoint, pick_device
    device = pick_device(device_pref)
    model, _ = load_checkpoint(ckpt_path, device=device)
    tok = CharTokenizer()

    def generate(prompt: str, max_new: int) -> str:
        return model.generate(tok.encode(prompt), tok, max_new, device=device)

    return generate, device, "scratch"


def rollout_task(generate, task: dict, rounds: int) -> dict:
    shapes = shapes_for(task["k"], task["h"])
    code = task["model_code"]
    cur = task["init_params"]
    # generation budget: a bit above the longest ground-truth target for this task
    max_new = max(len(build_target(r["trace"], r["params_out_text"]))
                  for r in task["gt_rollout"]) + 96
    recs = []
    for r in range(rounds):
        prompt = build_prompt(code, params_to_text(cur, shapes))
        gen = generate(prompt, max_new)
        trace, ptext, fmt_ok = parse_output(gen)
        params, parse_ok = parse_params_text(ptext, shapes, fallback=cur)
        recs.append({
            "round": r,
            "params_in_text": params_to_text(cur, shapes),
            "gen": gen,
            "trace": trace,
            "params_out_text": params_to_text(params, shapes),
            "format_ok": bool(fmt_ok and parse_ok),
        })
        cur = params
    return {"task_id": task["task_id"], "rounds": recs,
            "final_params": cur,
            "final_params_text": params_to_text(cur, shapes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help=".pt file (scratch) or HF model dir")
    ap.add_argument("--tasks", required=True, help="eval_tasks.json from ntp.datagen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None,
                    help="default: the rounds the eval set was generated with")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.tasks) as f:
        eval_set = json.load(f)
    tasks = eval_set["tasks"][:args.limit] if args.limit else eval_set["tasks"]
    rounds = args.rounds or eval_set["rounds"]

    generate, device, backend = make_generator(args.ckpt, args.device)
    print("backend=%s device=%s tasks=%d rounds=%d" % (backend, device, len(tasks), rounds))

    out = {"ckpt": args.ckpt, "backend": backend, "rounds": rounds, "tasks": []}
    t0 = time.time()
    for i, task in enumerate(tasks):
        ts = time.time()
        rec = rollout_task(generate, task, rounds)
        out["tasks"].append(rec)
        ok = sum(1 for r in rec["rounds"] if r["format_ok"])
        print("[%d/%d] %s  %.1fs  format_ok %d/%d" %
              (i + 1, len(tasks), task["task_id"], time.time() - ts, ok, rounds),
              flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s (%.1f min total)" % (args.out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
