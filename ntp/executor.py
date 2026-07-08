"""Ground-truth executor: run policy code for real, capture trace + updated params.

Parameters are quantized (via textio.qfloat) on the way in and out of every round, so
the executor computes exactly the text-level transition the model is trained to imitate.
"""
from __future__ import annotations

import contextlib
import copy
import io
from typing import Dict, List, Tuple

from .textio import Params, Shapes, params_to_text, parse_trace, quantize_params


def exec_code(code: str) -> dict:
    ns: dict = {}
    exec(compile(code, "<policy>", "exec"), ns)
    return ns


def run_round(code: str, params: Params, shapes: Shapes) -> Tuple[str, Params]:
    """Execute one policy_round. Returns (trace, quantized updated params)."""
    ns = exec_code(code)
    p = quantize_params(copy.deepcopy(params), shapes)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = ns["policy_round"](p)
    return buf.getvalue().strip(), quantize_params(out, shapes)


def rollout(code: str, params: Params, shapes: Shapes, rounds: int) -> List[dict]:
    """Run `rounds` consecutive rounds, threading quantized params through."""
    cur = quantize_params(copy.deepcopy(params), shapes)
    recs = []
    for r in range(rounds):
        trace, nxt = run_round(code, cur, shapes)
        recs.append({
            "round": r,
            "params_in": copy.deepcopy(cur),
            "params_in_text": params_to_text(cur, shapes),
            "trace": trace,
            "params_out": copy.deepcopy(nxt),
            "params_out_text": params_to_text(nxt, shapes),
        })
        cur = nxt
    return recs


def cluster_at(code: str, params: Params, shapes: Shapes) -> List[int]:
    """Hard cluster assignment under the given params (no training)."""
    ns = exec_code(code)
    p = quantize_params(copy.deepcopy(params), shapes)
    return list(ns["cluster"](p))


def loss_at(code: str, params: Params, shapes: Shapes) -> float:
    """Objective value at the given params = first printed loss of a round."""
    trace, _ = run_round(code, params, shapes)
    losses = parse_trace(trace)["losses"]
    if not losses:
        raise RuntimeError("no loss line in trace")
    return losses[0]
