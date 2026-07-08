"""Side-by-side inspection of one round: what the model predicted vs what the code
actually does from the model's own input state.

Usage:
    python3 -m ntp.compare --tasks data/v2/eval_tasks.json \
        --rollouts runs/v2_qwen05/rollouts.json --index 0 --round 0
"""
from __future__ import annotations

import argparse
import json

from .executor import run_round
from .textio import parse_params_text, shapes_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    with open(args.tasks) as f:
        eval_set = json.load(f)
    task = eval_set["tasks"][args.index]
    with open(args.rollouts) as f:
        ro = json.load(f)
    pred = next(t for t in ro["tasks"] if t["task_id"] == task["task_id"])
    r = pred["rounds"][args.round]

    shapes = shapes_for(task["k"], task["h"])
    p_in, _ = parse_params_text(r["params_in_text"], shapes, task["init_params"])
    true_trace, true_out = run_round(task["code"], p_in, shapes)

    print("# task %s  round %d  (K=%d H=%d LR=%.2f STEPS=%d)"
          % (task["task_id"], args.round, task["k"], task["h"], task["lr"], task["steps"]))
    print("\n--- params in (model state at this round) ---")
    print(r["params_in_text"])
    print("\n--- TRUE trace (real execution from that state) ---")
    print(true_trace)
    print("\n--- MODEL predicted trace ---")
    print(r["trace"] if r["trace"] else "(unparseable)")

    from .textio import params_to_text
    print("\n--- TRUE updated params ---")
    print(params_to_text(true_out, shapes))
    print("\n--- MODEL predicted updated params ---")
    print(r["params_out_text"])

    # quick per-line hint of where trace lines diverge
    tl, pl = true_trace.splitlines(), (r["trace"] or "").splitlines()
    print("\n--- line-level trace match ---")
    for i in range(max(len(tl), len(pl))):
        a = tl[i] if i < len(tl) else "(missing)"
        b = pl[i] if i < len(pl) else "(missing)"
        print("%s | %s" % ("OK  " if a == b else "DIFF", b if a == b else "true: %s   pred: %s" % (a, b)))


if __name__ == "__main__":
    main()
