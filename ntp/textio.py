"""Serialization layer: parameters <-> text, prompt/target construction, output parsing.

Everything the model reads or writes goes through this module, and the ground-truth
executor quantizes parameters through the same `qfloat` at every round boundary, so the
text-level mapping (code, params_text) -> (trace, params_text') is exactly well-defined.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

PARAM_DP = 3          # decimal places for parameters in text
DATA_DP = 2           # decimal places for data points
LOSS_DP = 4           # decimal places for printed losses
PARAM_MAX = 9.999     # clamp so every param fits a fixed-width field

PARAM_KEYS = ["W1", "b1", "W2", "b2"]

Params = Dict[str, list]
Shapes = Dict[str, tuple]

FLOAT_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def shapes_for(k: int, h: int) -> Shapes:
    return {"W1": (h, 2), "b1": (h,), "W2": (k, h), "b2": (k,)}


def qfloat(v: float, dp: int = PARAM_DP) -> float:
    """Quantize to dp decimals with clamping; the value a text round-trip yields."""
    v = max(-PARAM_MAX, min(PARAM_MAX, float(v)))
    q = round(v, dp)
    return 0.0 if q == 0.0 else q  # normalize -0.0


def fmt(v: float, dp: int = PARAM_DP) -> str:
    return ("%%.%df" % dp) % qfloat(v, dp)


def quantize_params(params: Params, shapes: Shapes) -> Params:
    out: Params = {}
    for key in PARAM_KEYS:
        shape = shapes[key]
        val = params[key]
        if len(shape) == 1:
            out[key] = [qfloat(v) for v in val]
        else:
            out[key] = [[qfloat(v) for v in row] for row in val]
    return out


def params_to_text(params: Params, shapes: Shapes) -> str:
    lines = []
    for key in PARAM_KEYS:
        shape = shapes[key]
        val = params[key]
        if len(shape) == 1:
            body = " ".join(fmt(v) for v in val)
        else:
            body = " / ".join(" ".join(fmt(v) for v in row) for row in val)
        lines.append("%s %s" % (key, body))
    return "\n".join(lines)


def _reshape(flat: List[float], shape: tuple, fallback_flat: List[float]) -> list:
    n = 1
    for d in shape:
        n *= d
    vals = list(flat[:n])
    while len(vals) < n:  # pad missing entries from fallback (i.e. "no update")
        vals.append(fallback_flat[len(vals)])
    vals = [qfloat(v) for v in vals]
    if len(shape) == 1:
        return vals
    rows, cols = shape
    return [vals[r * cols:(r + 1) * cols] for r in range(rows)]


def flatten_params(params: Params, shapes: Shapes) -> List[float]:
    flat: List[float] = []
    for key in PARAM_KEYS:
        val = params[key]
        if len(shapes[key]) == 1:
            flat.extend(val)
        else:
            for row in val:
                flat.extend(row)
    return flat


def parse_params_text(text: str, shapes: Shapes, fallback: Params) -> Tuple[Params, bool]:
    """Parse a params block robustly. Returns (params, exact_format_ok).

    Preferred path: one line per key, in order. Repair path: any missing/short keys are
    filled from `fallback` (the round's input params, i.e. "no update" for those entries).
    """
    per_key: Dict[str, List[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        head = line.split(None, 1)[0]
        if head in shapes and head not in per_key:
            nums = [float(m) for m in FLOAT_RE.findall(line[len(head):])]
            per_key[head] = nums

    ok = True
    out: Params = {}
    for key in PARAM_KEYS:
        shape = shapes[key]
        n = 1
        for d in shape:
            n *= d
        flat = per_key.get(key, [])
        if len(flat) != n:
            ok = False
        fb = fallback[key]
        fb_flat = list(fb) if len(shape) == 1 else [v for row in fb for v in row]
        out[key] = _reshape(flat, shape, fb_flat)
    return out, ok


# ---------------------------------------------------------------------------
# Prompt / target format
# ---------------------------------------------------------------------------

OUT_MARK = "<OUT>"
END_MARK = "<END>"


def build_prompt(code: str, params_text: str) -> str:
    return (
        "<CODE>\n" + code.strip() + "\n</CODE>\n"
        "<PARAMS>\n" + params_text.strip() + "\n</PARAMS>\n"
        + OUT_MARK + "\n"
    )


def build_target(trace: str, params_text: str) -> str:
    return (
        "<TRACE>\n" + trace.strip() + "\n</TRACE>\n"
        "<PARAMS>\n" + params_text.strip() + "\n</PARAMS>\n"
        + END_MARK
    )


TRACE_BLOCK_RE = re.compile(r"<TRACE>\n(.*?)</TRACE>", re.DOTALL)
PARAMS_BLOCK_RE = re.compile(r"<PARAMS>\n(.*?)</PARAMS>", re.DOTALL)


def parse_output(gen: str) -> Tuple[str, str, bool]:
    """Parse generated text into (trace, params_text, format_ok)."""
    if END_MARK in gen:
        gen = gen.split(END_MARK, 1)[0]
    trace_m = TRACE_BLOCK_RE.search(gen)
    params_m = PARAMS_BLOCK_RE.search(gen)
    trace = trace_m.group(1).strip() if trace_m else ""
    params_text = params_m.group(1).strip() if params_m else ""
    ok = trace_m is not None and params_m is not None
    if not params_m:
        # last resort: take everything after the last <PARAMS> or the tail of the text
        tail = gen.rsplit("<PARAMS>", 1)
        params_text = tail[1].strip() if len(tail) == 2 else ""
        ok = False
    return trace, params_text, ok


# ---------------------------------------------------------------------------
# Trace parsing (for metrics)
# ---------------------------------------------------------------------------

STEP_LOSS_RE = re.compile(r"step\s+(\d+)\s+loss\s+(-?\d+\.\d+)")
ASSIGN_RE = re.compile(r"assign\s+([0-9]+)")


def parse_trace(trace: str) -> Dict[str, object]:
    losses = [float(m[1]) for m in STEP_LOSS_RE.findall(trace)]
    am = ASSIGN_RE.search(trace)
    assign = [int(c) for c in am.group(1)] if am else []
    return {"losses": losses, "assign": assign}


def compact_code(code: str) -> str:
    """Strip comment lines / trailing comments / blank lines (template has no '#' in strings)."""
    out = []
    for line in code.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        if line.strip() == "":
            continue
        out.append(line)
    return "\n".join(out)
