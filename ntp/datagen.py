"""Generate training/val JSONL (one example per policy round) and eval task specs.

Each training example is a single-round transition:
    prompt = <CODE>code</CODE><PARAMS>params_r</PARAMS><OUT>
    target = <TRACE>trace_r</TRACE><PARAMS>params_{r+1}</PARAMS><END>
conditioned only on the current round's input (Markov), so at inference time the
model's own output can be fed back as the next input.

Usage:
    python3 -m ntp.datagen --out data/demo --train-tasks 3000 --val-tasks 100 \
        --eval-tasks 50 --rounds 4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .executor import rollout
from .metrics import ari, mean
from .tasks import SamplerConfig, TaskSpec, render_code, sample_task
from .textio import build_prompt, build_target, compact_code, parse_trace


def gen_split(rng: random.Random, n_tasks: int, rounds: int, cfg: SamplerConfig,
              prefix: str, use_compact: bool):
    """Yields (task_dict, examples) per task."""
    for i in range(n_tasks):
        spec = sample_task(rng, task_id="%s-%05d" % (prefix, i), cfg=cfg)
        code = render_code(spec)
        model_code = compact_code(code) if use_compact else code
        recs = rollout(code, spec.init_params, spec.shapes(), rounds)
        examples = []
        for r in recs:
            examples.append({
                "task_id": spec.task_id,
                "round": r["round"],
                "k": spec.k, "h": spec.h,
                "prompt": build_prompt(model_code, r["params_in_text"]),
                "target": build_target(r["trace"], r["params_out_text"]),
            })
        task = spec.to_json()
        task["code"] = code
        task["model_code"] = model_code
        task["gt_rollout"] = [
            {"round": r["round"], "params_in_text": r["params_in_text"],
             "trace": r["trace"], "params_out_text": r["params_out_text"]}
            for r in recs
        ]
        final_assign = parse_trace(recs[-1]["trace"])["assign"]
        task["gt_final_ari"] = ari(final_assign, spec.labels)
        yield task, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tasks", type=int, default=3000)
    ap.add_argument("--val-tasks", type=int, default=100)
    ap.add_argument("--eval-tasks", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-code", action="store_true",
                    help="feed the model the raw code (default: comment/blank-stripped)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = SamplerConfig()
    use_compact = not args.full_code
    lens = []

    for split, n_tasks, seed_off in (("train", args.train_tasks, 0),
                                     ("val", args.val_tasks, 1),
                                     ("eval", args.eval_tasks, 2)):
        rng = random.Random(args.seed * 1000 + seed_off)
        n_ex = 0
        gt_aris = []
        tasks_out = []
        with open(os.path.join(args.out, split + ".jsonl"), "w") as f:
            for task, examples in gen_split(rng, n_tasks, args.rounds, cfg, split, use_compact):
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
                    lens.append(len(ex["prompt"]) + len(ex["target"]))
                    n_ex += 1
                gt_aris.append(task["gt_final_ari"])
                if split == "eval":
                    tasks_out.append(task)
        if split == "eval":
            with open(os.path.join(args.out, "eval_tasks.json"), "w") as f:
                json.dump({"rounds": args.rounds, "tasks": tasks_out}, f, indent=1)
        print("%s: %d tasks, %d examples, GT final ARI mean %.3f (frac>=0.8: %.2f)"
              % (split, n_tasks, n_ex, mean(gt_aris),
                 sum(1 for a in gt_aris if a >= 0.8) / max(1, len(gt_aris))))

    lens.sort()
    print("example char length: p50=%d p95=%d max=%d"
          % (lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]))
    meta = {"rounds": args.rounds, "seed": args.seed, "compact_code": use_compact,
            "max_chars": lens[-1]}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
