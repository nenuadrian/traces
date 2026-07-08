"""Weights & Biases logging, None-safe everywhere.

Every pipeline stage logs to project $WANDB_PROJECT (default "neural-trace-policies"),
with runs grouped by experiment tag (e.g. all of v4_qwen05's train/dagger/infer/eval
runs share one group). Telemetry must never kill a run: init failures degrade to a
printed warning, and --no-wandb disables logging entirely.
"""
from __future__ import annotations

import os
from typing import Optional


def init_run(job_type: str, tag: str, config: dict, enabled: bool = True,
             name: Optional[str] = None):
    """Returns a wandb run or None. Never raises."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — logging disabled", flush=True)
        return None
    try:
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "neural-trace-policies"),
            name=name or "%s/%s" % (tag, job_type),
            group=tag, job_type=job_type, config=config)
    except Exception as e:  # offline queue full, no login, etc.
        print("wandb init failed (%s) — logging disabled" % e, flush=True)
        return None


def log(run, data: dict, step: Optional[int] = None):
    if run is None:
        return
    try:
        if step is None:
            run.log(data)
        else:
            run.log(data, step=step)
    except Exception:
        pass


def set_summary(run, data: dict):
    if run is None:
        return
    try:
        for k, v in data.items():
            run.summary[k] = v
    except Exception:
        pass


def finish(run):
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


def histogram(values):
    """wandb.Histogram or None if wandb unavailable."""
    try:
        import wandb
        return wandb.Histogram(values)
    except Exception:
        return None


def table(columns, rows):
    try:
        import wandb
        return wandb.Table(columns=columns, data=rows)
    except Exception:
        return None


def tag_from_out(path: str) -> str:
    """Experiment tag from an output path: runs/v4_qwen05/rollouts.json -> v4_qwen05."""
    path = os.path.abspath(path)
    base = os.path.basename(path)
    if "." in base:  # a file: use its directory name
        return os.path.basename(os.path.dirname(path)) or base
    return base


def data_meta(data_dir: str) -> dict:
    import json
    p = os.path.join(data_dir, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return {"data_" + k: v for k, v in json.load(f).items()}
    return {}
