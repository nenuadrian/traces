"""Iterative inference: the trained LM *is* the trainer.

For each eval task, start from the task's initial parameters and repeatedly feed
(code, current params) to the model; it emits (predicted trace, updated params — or a
<DELTA> update when the dataset was generated with --delta), and the result is fed
back in. After N rounds the final params are a model-trained policy — verify them with
ntp.evaluate / ntp.run_policy.

Generation is batched across tasks per round (HF backend), so wall time scales with
rounds, not tasks x rounds.

Usage:
    python3 -m ntp.infer --ckpt runs/v4/hf_model --tasks data/v4/eval_tasks.json \
        --out runs/v4/rollouts.json --rounds 12 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

from . import wb
from .textio import (apply_delta, build_prompt, build_target, params_to_text,
                     parse_delta_text, parse_output, parse_params_text, shapes_for)


def make_generator(ckpt_path: str, device_pref=None):
    """Returns (generate_batch(prompts, max_new) -> [str], device, backend_name)."""
    if os.path.isdir(ckpt_path):
        from .hf_backend import HFGenerator
        gen = HFGenerator(ckpt_path, device=device_pref)
        return gen.generate_batch, gen.device, "hf"
    from .scratch_model import CharTokenizer, load_checkpoint, pick_device
    device = pick_device(device_pref)
    model, _ = load_checkpoint(ckpt_path, device=device)
    tok = CharTokenizer()

    def generate_batch(prompts: List[str], max_new: int) -> List[str]:
        return [model.generate(tok.encode(p), tok, max_new, device=device)
                for p in prompts]

    return generate_batch, device, "scratch"


def target_budget(task: dict) -> int:
    """Generation budget: a bit above the longest ground-truth target for the task."""
    if not task.get("gt_rollout"):
        return task.get("_budget", 700)
    return max(len(build_target(r["trace"], r["params_out_text"]))
               for r in task["gt_rollout"]) + 128


def rollout_tasks(generate_batch, tasks: List[dict], rounds: int, delta: bool,
                  batch_size: int, run=None) -> List[dict]:
    shapes_l = [shapes_for(t["k"], t["h"]) for t in tasks]
    states = [t["init_params"] for t in tasks]
    recs = [[] for _ in tasks]
    budgets = [target_budget(t) for t in tasks]

    for r in range(rounds):
        t0 = time.time()
        prompts = [build_prompt(t["model_code"], params_to_text(states[i], shapes_l[i]))
                   for i, t in enumerate(tasks)]
        gens: List[str] = []
        for c in range(0, len(prompts), batch_size):
            gens.extend(generate_batch(prompts[c:c + batch_size],
                                       max(budgets[c:c + batch_size])))
        n_ok = 0
        for i in range(len(tasks)):
            trace, block, fmt_ok = parse_output(gens[i], delta=delta)
            if delta:
                d, parse_ok = parse_delta_text(block, shapes_l[i])
                params = apply_delta(states[i], d, shapes_l[i])
            else:
                params, parse_ok = parse_params_text(block, shapes_l[i],
                                                     fallback=states[i])
            ok = bool(fmt_ok and parse_ok)
            n_ok += 1 if ok else 0
            recs[i].append({
                "round": r,
                "params_in_text": params_to_text(states[i], shapes_l[i]),
                "gen": gens[i],
                "trace": trace,
                "params_out_text": params_to_text(params, shapes_l[i]),
                "format_ok": ok,
            })
            states[i] = params
        print("[round %d/%d] %.1fs  format_ok %d/%d"
              % (r + 1, rounds, time.time() - t0, n_ok, len(tasks)), flush=True)
        wb.log(run, {"infer/format_ok_rate": n_ok / len(tasks),
                     "infer/round_sec": time.time() - t0}, step=r + 1)

    return [{"task_id": t["task_id"], "rounds": recs[i],
             "final_params": states[i],
             "final_params_text": params_to_text(states[i], shapes_l[i])}
            for i, t in enumerate(tasks)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help=".pt file (scratch) or HF model dir")
    ap.add_argument("--tasks", required=True, help="eval_tasks.json from ntp.datagen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None,
                    help="default: the rounds the eval set was generated with")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(args.tasks) as f:
        eval_set = json.load(f)
    tasks = eval_set["tasks"][:args.limit] if args.limit else eval_set["tasks"]
    rounds = args.rounds or eval_set["rounds"]
    delta = bool(eval_set.get("delta", False))

    generate_batch, device, backend = make_generator(args.ckpt, args.device)
    print("backend=%s device=%s tasks=%d rounds=%d delta=%s batch=%d"
          % (backend, device, len(tasks), rounds, delta, args.batch_size))
    run = wb.init_run("infer", wb.tag_from_out(args.out), dict(
        vars(args), backend=backend, device=device, n_tasks=len(tasks),
        rounds=rounds, delta=delta), enabled=not args.no_wandb)

    t0 = time.time()
    task_recs = rollout_tasks(generate_batch, tasks, rounds, delta, args.batch_size,
                              run=run)
    out = {"ckpt": args.ckpt, "backend": backend, "rounds": rounds, "delta": delta,
           "tasks": task_recs}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s (%.1f min total)" % (args.out, (time.time() - t0) / 60))
    all_ok = [r["format_ok"] for t in task_recs for r in t["rounds"]]
    wb.set_summary(run, {"format_ok": sum(all_ok) / max(1, len(all_ok)),
                         "total_min": (time.time() - t0) / 60})
    wb.finish(run)


if __name__ == "__main__":
    main()
