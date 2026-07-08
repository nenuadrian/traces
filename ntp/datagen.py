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

from .executor import rollout, run_round
from .metrics import ari, mean
from .tasks import SamplerConfig, TaskSpec, render_code, sample_task
from .textio import (PARAM_KEYS, build_prompt, build_target, build_target_delta,
                     compact_code, delta_params, delta_to_text, params_to_text,
                     parse_trace, quantize_params)


def make_target(p_in, p_out, trace, shapes, delta_mode: bool) -> str:
    if delta_mode:
        return build_target_delta(trace, delta_to_text(delta_params(p_in, p_out, shapes), shapes))
    return build_target(trace, params_to_text(p_out, shapes))


def _jitter_params(params, shapes, rng: random.Random, sigma: float):
    out = {}
    for key in PARAM_KEYS:
        v = params[key]
        if len(shapes[key]) == 1:
            out[key] = [x + rng.gauss(0.0, sigma) for x in v]
        else:
            out[key] = [[x + rng.gauss(0.0, sigma) for x in row] for row in v]
    return quantize_params(out, shapes)


def gen_split(rng: random.Random, n_tasks: int, rounds: int, cfg: SamplerConfig,
              prefix: str, use_compact: bool,
              jitter_frac: float = 0.0, jitter_sigmas=(0.08,), delta_mode: bool = False):
    """Yields (task_dict, examples) per task.

    jitter_frac > 0 adds DAgger-style examples: a round's input params are perturbed
    with Gaussian noise and the round is re-executed from there, teaching the model to
    descend from imperfect (i.e. its own, slightly-off) states, not only from exact
    ground-truth trajectories.
    """
    for i in range(n_tasks):
        spec = sample_task(rng, task_id="%s-%05d" % (prefix, i), cfg=cfg)
        code = render_code(spec)
        model_code = compact_code(code) if use_compact else code
        shapes = spec.shapes()
        recs = rollout(code, spec.init_params, shapes, rounds)
        examples = []
        for r in recs:
            examples.append({
                "task_id": spec.task_id,
                "round": r["round"],
                "k": spec.k, "h": spec.h,
                "prompt": build_prompt(model_code, r["params_in_text"]),
                "target": make_target(r["params_in"], r["params_out"], r["trace"],
                                      shapes, delta_mode),
            })
        for r in recs:
            if jitter_frac > 0.0 and rng.random() < jitter_frac:
                jp = _jitter_params(r["params_in"], shapes, rng,
                                    rng.choice(list(jitter_sigmas)))
                trace_j, out_j = run_round(code, jp, shapes)
                examples.append({
                    "task_id": spec.task_id,
                    "round": r["round"], "jitter": True,
                    "k": spec.k, "h": spec.h,
                    "prompt": build_prompt(model_code, params_to_text(jp, shapes)),
                    "target": make_target(jp, out_j, trace_j, shapes, delta_mode),
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
    ap.add_argument("--jitter-frac", type=float, default=0.0,
                    help="fraction of rounds to also emit from noise-perturbed params "
                         "(train/val only; DAgger-style robustness to the model's own drift)")
    ap.add_argument("--jitter-sigma", default="0.08",
                    help="noise scale, or comma-separated scales sampled per example "
                         "(e.g. '0.05,0.15,0.3')")
    ap.add_argument("--delta", action="store_true",
                    help="targets carry <DELTA> (signed param updates) instead of "
                         "absolute <PARAMS> — recommended; removes the copy-bias")
    ap.add_argument("--dup-early", type=int, default=1,
                    help="write rounds 0-1 examples this many times (train/val only): "
                         "big-update rounds carry the magnitude signal the model "
                         "otherwise regresses away")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    jitter_sigmas = [float(s) for s in str(args.jitter_sigma).split(",") if s]

    from . import wb
    run = wb.init_run("datagen", wb.tag_from_out(args.out), vars(args),
                      enabled=not args.no_wandb)

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
        jf = args.jitter_frac if split != "eval" else 0.0
        dup = args.dup_early if split != "eval" else 1
        with open(os.path.join(args.out, split + ".jsonl"), "w") as f:
            for task, examples in gen_split(rng, n_tasks, args.rounds, cfg, split,
                                            use_compact, jf, jitter_sigmas, args.delta):
                for ex in examples:
                    reps = dup if ex["round"] <= 1 else 1
                    for _ in range(reps):
                        f.write(json.dumps(ex) + "\n")
                        n_ex += 1
                    lens.append(len(ex["prompt"]) + len(ex["target"]))
                gt_aris.append(task["gt_final_ari"])
                if split == "eval":
                    tasks_out.append(task)
        if split == "eval":
            with open(os.path.join(args.out, "eval_tasks.json"), "w") as f:
                json.dump({"rounds": args.rounds, "delta": args.delta,
                           "tasks": tasks_out}, f, indent=1)
        print("%s: %d tasks, %d examples, GT final ARI mean %.3f (frac>=0.8: %.2f)"
              % (split, n_tasks, n_ex, mean(gt_aris),
                 sum(1 for a in gt_aris if a >= 0.8) / max(1, len(gt_aris))))
        wb.set_summary(run, {split + "/tasks": n_tasks, split + "/examples": n_ex,
                             split + "/gt_ari_mean": mean(gt_aris)})
        h = wb.histogram(gt_aris)
        if h is not None:
            wb.log(run, {"dist/gt_final_ari_" + split: h})

    lens.sort()
    print("example char length: p50=%d p95=%d max=%d"
          % (lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]))
    meta = {"rounds": args.rounds, "seed": args.seed, "compact_code": use_compact,
            "delta": args.delta, "jitter_frac": args.jitter_frac,
            "jitter_sigmas": jitter_sigmas, "max_chars": lens[-1]}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    wb.set_summary(run, {"chars_p50": lens[len(lens) // 2], "chars_max": lens[-1]})
    wb.finish(run)


if __name__ == "__main__":
    main()
