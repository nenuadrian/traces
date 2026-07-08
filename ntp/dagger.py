"""True DAgger round: let the trained model roll out on *training* tasks, collect the
states it actually visits (including its drifted/plateau states), execute the real code
from those states, and emit the corrections as extra training examples.

Continue training on base + these examples:
    python3 -m ntp.dagger --ckpt runs/v4/hf_model --data data/v4 \
        --out data/v4/dagger.jsonl --tasks 1000 --rounds 6
    python3 -m ntp.train --backend hf --hf-model runs/v4/hf_model --data data/v4 \
        --extra-train data/v4/dagger.jsonl --out runs/v4_dagger ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from . import wb
from .datagen import make_target
from .executor import run_round
from .infer import make_generator, rollout_tasks
from .textio import (build_prompt, parse_params_text, parse_trace, shapes_for,
                     zeros_like_shapes)


def tasks_from_train_jsonl(path: str, n: int, rng: random.Random):
    """Reconstruct runnable pseudo-tasks from round-0 training prompts (the compact
    code embedded in each prompt is itself a complete, executable program)."""
    seeds = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["round"] == 0 and not ex.get("jitter"):
                seeds.append(ex)
    rng.shuffle(seeds)
    tasks = []
    for ex in seeds[:n]:
        code = ex["prompt"].split("<CODE>\n", 1)[1].split("\n</CODE>", 1)[0]
        ptext = ex["prompt"].split("<PARAMS>\n", 1)[1].split("\n</PARAMS>", 1)[0]
        shapes = shapes_for(ex["k"], ex["h"])
        params, ok = parse_params_text(ptext, shapes, zeros_like_shapes(shapes))
        assert ok, "could not parse params from training prompt"
        tasks.append({
            "task_id": ex["task_id"], "k": ex["k"], "h": ex["h"],
            "model_code": code, "init_params": params,
            "_budget": len(ex["target"]) + 128,
        })
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset dir (train.jsonl + meta.json)")
    ap.add_argument("--out", required=True, help="output jsonl of corrective examples")
    ap.add_argument("--tasks", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    delta = bool(meta.get("delta", False))
    rng = random.Random(args.seed)
    tasks = tasks_from_train_jsonl(os.path.join(args.data, "train.jsonl"),
                                   args.tasks, rng)
    generate_batch, device, backend = make_generator(args.ckpt, args.device)
    print("backend=%s device=%s tasks=%d rounds=%d delta=%s"
          % (backend, device, len(tasks), args.rounds, delta), flush=True)
    ck = os.path.abspath(args.ckpt)
    tag = os.path.basename(os.path.dirname(ck)) if os.path.isdir(ck) \
        else wb.tag_from_out(ck)
    run = wb.init_run("dagger", tag, dict(vars(args), backend=backend, device=device,
                                          delta=delta, n_tasks=len(tasks)),
                      enabled=not args.no_wandb)

    t0 = time.time()
    recs = rollout_tasks(generate_batch, tasks, args.rounds, delta, args.batch_size,
                         run=run)

    n_ex = 0
    visited_losses = []
    with open(args.out, "w") as f:
        for task, rec in zip(tasks, recs):
            shapes = shapes_for(task["k"], task["h"])
            # every state the model produced (params_out of each round) gets a
            # ground-truth correction from the real executor
            seen = set()
            for r in rec["rounds"]:
                st_text = r["params_out_text"]
                if st_text in seen:
                    continue
                seen.add(st_text)
                state, _ = parse_params_text(st_text, shapes,
                                             zeros_like_shapes(shapes))
                trace, out = run_round(task["model_code"], state, shapes)
                visited_losses.append(parse_trace(trace)["losses"][0])
                f.write(json.dumps({
                    "task_id": task["task_id"], "round": r["round"], "dagger": True,
                    "k": task["k"], "h": task["h"],
                    "prompt": build_prompt(task["model_code"], st_text),
                    "target": make_target(state, out, trace, shapes, delta),
                }) + "\n")
                n_ex += 1
    print("wrote %d corrective examples -> %s (%.1f min)"
          % (n_ex, args.out, (time.time() - t0) / 60))
    wb.set_summary(run, {"n_examples": n_ex, "total_min": (time.time() - t0) / 60,
                         "visited_loss_mean": sum(visited_losses) / max(1, len(visited_losses))})
    h = wb.histogram(visited_losses)
    if h is not None:
        wb.log(run, {"dist/visited_state_loss": h})
    wb.finish(run)


if __name__ == "__main__":
    main()
