"""Evaluate model rollouts against ground truth — including by *running the real code*
with the model-predicted parameters.

Metrics
  format_ok        fraction of rounds whose output parsed cleanly
  onestep/*        one-round simulation fidelity, teacher-forced on the model's own
                   current state: re-run the real code from the model's round input and
                   compare its true trace/params to what the model predicted
    loss_mae         |predicted loss lines - true loss lines|
    assign_acc       fraction of per-point cluster assignments predicted correctly
    param_mae        |predicted updated params - true updated params|
  openloop_param_mae_r  drift of the iterated model params vs the GT training run
  selfcons_assign_acc   run the real code with the model's FINAL params: does its actual
                        clustering match what the model claimed in its final trace?
  quality/*        ARI vs the true blob labels and objective value, for: init params,
                   GT-trained params, and model-trained params
Sanity check the harness with --oracle (uses GT rollouts as "predictions"; all
fidelity metrics should be perfect).

Usage:
    python3 -m ntp.evaluate --tasks data/demo/eval_tasks.json --rollouts runs/demo/rollouts.json
"""
from __future__ import annotations

import argparse
import json
from typing import List

from .executor import cluster_at, loss_at, run_round
from .metrics import ari, mae, mean, median
from .textio import (flatten_params, parse_params_text, parse_trace, shapes_for)


def _params_from_text(text: str, shapes, fallback):
    p, _ = parse_params_text(text, shapes, fallback)
    return p


def eval_task(task: dict, pred: dict) -> dict:
    shapes = shapes_for(task["k"], task["h"])
    code = task["code"]
    init = task["init_params"]
    gt = task["gt_rollout"]
    rounds = pred["rounds"]

    m = {"task_id": task["task_id"], "format_ok": mean([1.0 if r["format_ok"] else 0.0
                                                        for r in rounds])}

    # --- one-step fidelity (teacher-forced on the model's own state) ---
    loss_maes, assign_accs, p_maes = [], [], []
    for r in rounds:
        p_in = _params_from_text(r["params_in_text"], shapes, init)
        true_trace, true_out = run_round(code, p_in, shapes)
        tp, pp = parse_trace(true_trace), parse_trace(r["trace"])
        if pp["losses"] and len(pp["losses"]) == len(tp["losses"]):
            loss_maes.append(mae(pp["losses"], tp["losses"]))
        else:
            loss_maes.append(float("nan"))
        if pp["assign"] and len(pp["assign"]) == len(tp["assign"]):
            assign_accs.append(sum(1 for a, b in zip(pp["assign"], tp["assign"]) if a == b)
                               / len(tp["assign"]))
        else:
            assign_accs.append(0.0)
        pred_out = _params_from_text(r["params_out_text"], shapes, p_in)
        p_maes.append(mae(flatten_params(pred_out, shapes), flatten_params(true_out, shapes)))
    m["onestep_loss_mae"] = mean([x for x in loss_maes if x == x])
    m["onestep_assign_acc"] = mean(assign_accs)
    m["onestep_param_mae"] = mean(p_maes)

    # --- open-loop drift vs the GT training run ---
    m["openloop_param_mae"] = []
    for r, g in zip(rounds, gt):
        pred_out = _params_from_text(r["params_out_text"], shapes, init)
        gt_out = _params_from_text(g["params_out_text"], shapes, init)
        m["openloop_param_mae"].append(mae(flatten_params(pred_out, shapes),
                                           flatten_params(gt_out, shapes)))

    # --- self-consistency: run the real code with the model's final params ---
    final_params = _params_from_text(pred["final_params_text"], shapes, init)
    real_assign = cluster_at(code, final_params, shapes)
    claimed = parse_trace(rounds[-1]["trace"])["assign"]
    if claimed and len(claimed) == len(real_assign):
        m["selfcons_assign_acc"] = sum(1 for a, b in zip(claimed, real_assign) if a == b) \
            / len(real_assign)
    else:
        m["selfcons_assign_acc"] = 0.0

    # --- quality of the model-trained policy ---
    labels = task["labels"]
    gt_final = _params_from_text(gt[-1]["params_out_text"], shapes, init)
    m["ari_init"] = ari(cluster_at(code, init, shapes), labels)
    m["ari_gt"] = ari(cluster_at(code, gt_final, shapes), labels)
    m["ari_model"] = ari(real_assign, labels)
    m["loss_init"] = loss_at(code, init, shapes)
    m["loss_gt"] = loss_at(code, gt_final, shapes)
    m["loss_model"] = loss_at(code, final_params, shapes)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--rollouts", help="rollouts.json from ntp.infer")
    ap.add_argument("--oracle", action="store_true",
                    help="evaluate GT rollouts against themselves (harness sanity check)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.tasks) as f:
        eval_set = json.load(f)
    tasks = {t["task_id"]: t for t in eval_set["tasks"]}

    if args.oracle:
        preds = []
        for t in eval_set["tasks"]:
            preds.append({
                "task_id": t["task_id"],
                "rounds": [{"round": g["round"], "params_in_text": g["params_in_text"],
                            "trace": g["trace"], "params_out_text": g["params_out_text"],
                            "format_ok": True} for g in t["gt_rollout"]],
                "final_params_text": t["gt_rollout"][-1]["params_out_text"],
            })
    else:
        if not args.rollouts:
            ap.error("--rollouts required unless --oracle")
        with open(args.rollouts) as f:
            preds = json.load(f)["tasks"]

    per_task = [eval_task(tasks[p["task_id"]], p) for p in preds]
    n_rounds = max(len(t["openloop_param_mae"]) for t in per_task)

    def agg(key, fn=mean):
        return fn([t[key] for t in per_task if t[key] == t[key]])

    summary = {
        "n_tasks": len(per_task),
        "format_ok": agg("format_ok"),
        "onestep_loss_mae": agg("onestep_loss_mae"),
        "onestep_assign_acc": agg("onestep_assign_acc"),
        "onestep_param_mae": agg("onestep_param_mae"),
        "openloop_param_mae_by_round": [
            mean([t["openloop_param_mae"][r] for t in per_task
                  if len(t["openloop_param_mae"]) > r]) for r in range(n_rounds)],
        "selfcons_assign_acc": agg("selfcons_assign_acc"),
        "ari_init": agg("ari_init"),
        "ari_gt": agg("ari_gt"),
        "ari_model": agg("ari_model"),
        "ari_model_median": agg("ari_model", median),
        "loss_init": agg("loss_init"),
        "loss_gt": agg("loss_gt"),
        "loss_model": agg("loss_model"),
    }

    print("== %d tasks ==" % summary["n_tasks"])
    print("format_ok:                %.3f" % summary["format_ok"])
    print("one-step fidelity (teacher-forced on model state):")
    print("  loss MAE:               %.4f" % summary["onestep_loss_mae"])
    print("  assign acc:             %.3f" % summary["onestep_assign_acc"])
    print("  param MAE:              %.4f" % summary["onestep_param_mae"])
    print("open-loop param MAE by round: %s"
          % " ".join("%.4f" % v for v in summary["openloop_param_mae_by_round"]))
    print("self-consistency (real run of code w/ model params vs model's claim):")
    print("  assign acc:             %.3f" % summary["selfcons_assign_acc"])
    print("policy quality (objective + ARI vs true blob labels):")
    print("  loss:  init %.3f  ->  GT-trained %.3f  |  model-trained %.3f"
          % (summary["loss_init"], summary["loss_gt"], summary["loss_model"]))
    print("  ARI:   init %.3f  ->  GT-trained %.3f  |  model-trained %.3f (median %.3f)"
          % (summary["ari_init"], summary["ari_gt"], summary["ari_model"],
             summary["ari_model_median"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_task": per_task}, f, indent=1)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
