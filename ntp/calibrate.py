"""Sanity-check the task family: does GT training converge, stay bounded, cluster well?

Usage: python3 -m ntp.calibrate --n 60 --rounds 4
"""
from __future__ import annotations

import argparse
import random

from .executor import rollout
from .metrics import ari, mean, median
from .tasks import SamplerConfig, render_code, sample_task
from .textio import flatten_params, parse_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = SamplerConfig()
    first_losses, final_losses, aris, maxabs, drifts = [], [], [], [], []
    for i in range(args.n):
        spec = sample_task(rng, "cal-%03d" % i, cfg)
        code = render_code(spec)
        recs = rollout(code, spec.init_params, spec.shapes(), args.rounds)
        losses = []
        for r in recs:
            losses.extend(parse_trace(r["trace"])["losses"])
        first_losses.append(losses[0])
        final_losses.append(losses[-1])
        fin = parse_trace(recs[-1]["trace"])["assign"]
        aris.append(ari(fin, spec.labels))
        flat0 = flatten_params(recs[0]["params_in"], spec.shapes())
        flatT = flatten_params(recs[-1]["params_out"], spec.shapes())
        maxabs.append(max(abs(v) for v in flatT))
        drifts.append(mean([abs(a - b) for a, b in zip(flat0, flatT)]))

    print("tasks: %d, rounds: %d" % (args.n, args.rounds))
    print("loss:  start mean %.3f -> final mean %.3f (median %.3f -> %.3f)"
          % (mean(first_losses), mean(final_losses), median(first_losses), median(final_losses)))
    print("ARI:   mean %.3f  median %.3f  frac>=0.8 %.2f  frac>=0.5 %.2f"
          % (mean(aris), median(aris),
             sum(1 for a in aris if a >= 0.8) / args.n,
             sum(1 for a in aris if a >= 0.5) / args.n))
    print("param: max|v| max %.3f  mean per-param |drift| %.4f"
          % (max(maxabs), mean(drifts)))


if __name__ == "__main__":
    main()
