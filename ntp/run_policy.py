"""Run a task's real policy code yourself, with any parameters — e.g. the ones the
model produced — and see the actual trace / clustering.

Examples:
    # actual behavior of the model-trained parameters for eval task 0
    python3 -m ntp.run_policy --tasks data/demo/eval_tasks.json --index 0 \
        --rollouts runs/demo/rollouts.json

    # ground-truth training from the initial parameters
    python3 -m ntp.run_policy --tasks data/demo/eval_tasks.json --index 0 --init --rounds 4
"""
from __future__ import annotations

import argparse
import json

from .executor import cluster_at, loss_at, rollout
from .metrics import ari
from .textio import parse_params_text, shapes_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--index", type=int, default=0)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollouts", help="use the model's final params from this rollouts.json")
    src.add_argument("--params-file", help="text file in the <PARAMS> block format")
    src.add_argument("--init", action="store_true", help="use the task's initial params")
    ap.add_argument("--rounds", type=int, default=0,
                    help="additionally run this many real training rounds from those params")
    args = ap.parse_args()

    with open(args.tasks) as f:
        eval_set = json.load(f)
    task = eval_set["tasks"][args.index]
    shapes = shapes_for(task["k"], task["h"])
    code = task["code"]

    if args.rollouts:
        with open(args.rollouts) as f:
            ro = json.load(f)
        rec = next(t for t in ro["tasks"] if t["task_id"] == task["task_id"])
        params, _ = parse_params_text(rec["final_params_text"], shapes, task["init_params"])
        print("# params: model-trained (final round output) for %s" % task["task_id"])
    elif args.params_file:
        with open(args.params_file) as f:
            params, ok = parse_params_text(f.read(), shapes, task["init_params"])
        print("# params: %s (parse ok: %s)" % (args.params_file, ok))
    else:
        params = task["init_params"]
        print("# params: task initial params")

    print("# task %s | K=%d H=%d LR=%.2f STEPS=%d N=%d"
          % (task["task_id"], task["k"], task["h"], task["lr"], task["steps"],
             len(task["data"])))
    assign = cluster_at(code, params, shapes)
    print("objective loss: %.4f" % loss_at(code, params, shapes))
    print("assignments:    %s" % "".join(str(a) for a in assign))
    print("ARI vs labels:  %.3f" % ari(assign, task["labels"]))

    if args.rounds:
        print("\n# running %d real training round(s) from these params:" % args.rounds)
        recs = rollout(code, params, shapes, args.rounds)
        for r in recs:
            print("--- round %d ---" % r["round"])
            print(r["trace"])
        print("final params:")
        print(recs[-1]["params_out_text"])


if __name__ == "__main__":
    main()
