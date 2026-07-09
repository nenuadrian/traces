"""Task family: pure-Python clustering policies driven by a tiny MLP.

Each task is a standalone Python program (no imports beyond `math`) that defines:
  cluster(params)       -> hard cluster assignment for every data point
  policy_round(params)  -> runs STEPS manual-backprop SGD steps on an EM-style
                           soft k-means objective, prints a trace, returns params

The dataset (2D points) is embedded in the code as a literal, so code + params is a
complete, runnable artifact. Gradients are hand-derived (centroids treated as constants
per step, i.e. an EM-flavored update), which keeps the program dependency-free and
deterministic.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .textio import DATA_DP, qfloat, quantize_params, shapes_for

TEMPLATE = '''import math

K = {k}
H = {h}
LR = {lr}
STEPS = {steps}

DATA = [
{data_rows}
]

def forward(params, pt):
    x, y = pt
    h = [math.tanh(params["W1"][j][0]*x + params["W1"][j][1]*y + params["b1"][j]) for j in range(H)]
    z = [sum(params["W2"][k][j]*h[j] for j in range(H)) + params["b2"][k] for k in range(K)]
    return h, z

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def cluster(params):
    return [max(range(K), key=lambda k: forward(params, pt)[1][k]) for pt in DATA]

def policy_round(params):
    n = len(DATA)
    for step in range(STEPS):
        HZ = [forward(params, pt) for pt in DATA]
        P = [softmax(z) for h, z in HZ]
        cw = [sum(p[k] for p in P) + 1e-9 for k in range(K)]
        C = [(sum(p[k]*pt[0] for pt, p in zip(DATA, P))/cw[k], sum(p[k]*pt[1] for pt, p in zip(DATA, P))/cw[k]) for k in range(K)]
        D = [[(pt[0]-C[k][0])**2 + (pt[1]-C[k][1])**2 for k in range(K)] for pt in DATA]
        S = [sum(P[i][k]*D[i][k] for k in range(K)) for i in range(n)]
        loss = sum(S) / n
        GZ = [[P[i][k]*(D[i][k]-S[i])/n for k in range(K)] for i in range(n)]
        GH = [[sum(GZ[i][k]*params["W2"][k][j] for k in range(K))*(1.0-HZ[i][0][j]**2) for j in range(H)] for i in range(n)]
        for k in range(K):
            params["b2"][k] -= LR*sum(GZ[i][k] for i in range(n))
            for j in range(H):
                params["W2"][k][j] -= LR*sum(GZ[i][k]*HZ[i][0][j] for i in range(n))
        for j in range(H):
            params["W1"][j][0] -= LR*sum(GH[i][j]*DATA[i][0] for i in range(n))
            params["W1"][j][1] -= LR*sum(GH[i][j]*DATA[i][1] for i in range(n))
            params["b1"][j] -= LR*sum(GH[i][j] for i in range(n))
        print("w " + " ".join("%.1f" % cw[k] for k in range(K)))
        print("c " + " / ".join("%.3f %.3f" % C[k] for k in range(K)))
        print("step %d loss %.4f" % (step + 1, loss))
    a = cluster(params)
    print("assign " + "".join(str(v) for v in a))
    print("counts " + " ".join(str(a.count(k)) for k in range(K)))
    return params
'''

# Gradient-trace variant: identical dynamics, but each step prints the per-parameter
# update (-LR*grad) it is about to apply, as g-blocks, before applying it. The round's
# net <DELTA> is then the sum of the printed updates — the model computes backprop on
# the page instead of in-weights. (Trace is longer; use with the HF backend.)
TEMPLATE_GRAD = '''import math

K = {k}
H = {h}
LR = {lr}
STEPS = {steps}

DATA = [
{data_rows}
]

def forward(params, pt):
    x, y = pt
    h = [math.tanh(params["W1"][j][0]*x + params["W1"][j][1]*y + params["b1"][j]) for j in range(H)]
    z = [sum(params["W2"][k][j]*h[j] for j in range(H)) + params["b2"][k] for k in range(K)]
    return h, z

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def cluster(params):
    return [max(range(K), key=lambda k: forward(params, pt)[1][k]) for pt in DATA]

def policy_round(params):
    n = len(DATA)
    for step in range(STEPS):
        HZ = [forward(params, pt) for pt in DATA]
        P = [softmax(z) for h, z in HZ]
        cw = [sum(p[k] for p in P) + 1e-9 for k in range(K)]
        C = [(sum(p[k]*pt[0] for pt, p in zip(DATA, P))/cw[k], sum(p[k]*pt[1] for pt, p in zip(DATA, P))/cw[k]) for k in range(K)]
        D = [[(pt[0]-C[k][0])**2 + (pt[1]-C[k][1])**2 for k in range(K)] for pt in DATA]
        S = [sum(P[i][k]*D[i][k] for k in range(K)) for i in range(n)]
        loss = sum(S) / n
        GZ = [[P[i][k]*(D[i][k]-S[i])/n for k in range(K)] for i in range(n)]
        GH = [[sum(GZ[i][k]*params["W2"][k][j] for k in range(K))*(1.0-HZ[i][0][j]**2) for j in range(H)] for i in range(n)]
        uW2 = [[-LR*sum(GZ[i][k]*HZ[i][0][j] for i in range(n)) for j in range(H)] for k in range(K)]
        ub2 = [-LR*sum(GZ[i][k] for i in range(n)) for k in range(K)]
        uW1 = [[-LR*sum(GH[i][j]*DATA[i][0] for i in range(n)), -LR*sum(GH[i][j]*DATA[i][1] for i in range(n))] for j in range(H)]
        ub1 = [-LR*sum(GH[i][j] for i in range(n)) for j in range(H)]
        print("w " + " ".join("%.1f" % cw[k] for k in range(K)))
        print("c " + " / ".join("%.3f %.3f" % C[k] for k in range(K)))
        print("gW1 " + " / ".join("%+.3f %+.3f" % (uW1[j][0], uW1[j][1]) for j in range(H)))
        print("gb1 " + " ".join("%+.3f" % v for v in ub1))
        print("gW2 " + " / ".join(" ".join("%+.3f" % uW2[k][j] for j in range(H)) for k in range(K)))
        print("gb2 " + " ".join("%+.3f" % v for v in ub2))
        for k in range(K):
            params["b2"][k] += ub2[k]
            for j in range(H):
                params["W2"][k][j] += uW2[k][j]
        for j in range(H):
            params["W1"][j][0] += uW1[j][0]
            params["W1"][j][1] += uW1[j][1]
            params["b1"][j] += ub1[j]
        print("step %d loss %.4f" % (step + 1, loss))
    a = cluster(params)
    print("assign " + "".join(str(v) for v in a))
    print("counts " + " ".join(str(a.count(k)) for k in range(K)))
    return params
'''


@dataclass
class TaskSpec:
    task_id: str
    k: int
    h: int
    lr: float
    steps: int
    data: List[Tuple[float, float]]
    labels: List[int]                  # generating blob id per point (never shown to model)
    init_params: Dict[str, list]
    grad_trace: bool = False           # print per-step gradient updates in the trace

    def shapes(self):
        return shapes_for(self.k, self.h)

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id, "k": self.k, "h": self.h, "lr": self.lr,
            "steps": self.steps, "data": self.data, "labels": self.labels,
            "init_params": self.init_params, "grad_trace": self.grad_trace,
        }

    @staticmethod
    def from_json(d: dict) -> "TaskSpec":
        return TaskSpec(
            task_id=d["task_id"], k=d["k"], h=d["h"], lr=d["lr"], steps=d["steps"],
            data=[tuple(p) for p in d["data"]], labels=d["labels"],
            init_params=d["init_params"], grad_trace=d.get("grad_trace", False),
        )


def render_code(spec: TaskSpec) -> str:
    rows = []
    for i in range(0, len(spec.data), 4):
        chunk = spec.data[i:i + 4]
        rows.append("    " + " ".join(
            "(%s, %s)," % (("%%.%df" % DATA_DP) % p[0], ("%%.%df" % DATA_DP) % p[1])
            for p in chunk))
    template = TEMPLATE_GRAD if spec.grad_trace else TEMPLATE
    return template.format(
        k=spec.k, h=spec.h, lr=("%.2f" % spec.lr), steps=spec.steps,
        data_rows="\n".join(rows))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    # Calibrated so ground-truth training reliably converges within ~4 rounds
    # (see ntp/calibrate.py): ARI>=0.8 on ~90% of tasks, params stay in [-5, 5].
    k_choices: Tuple[int, ...] = (2, 3)
    h_choices: Tuple[int, ...] = (4,)
    steps_choices: Tuple[int, ...] = (3,)
    lr_range: Tuple[float, float] = (0.60, 1.20)
    n_range: Tuple[int, int] = (14, 18)
    center_box: float = 2.3
    min_center_dist: float = 2.4
    sigma_range: Tuple[float, float] = (0.20, 0.30)
    init_scale: float = 1.0


def _sample_centers(rng: random.Random, k: int, cfg: SamplerConfig) -> List[Tuple[float, float]]:
    while True:
        cs = [(rng.uniform(-cfg.center_box, cfg.center_box),
               rng.uniform(-cfg.center_box, cfg.center_box)) for _ in range(k)]
        ok = True
        for i in range(k):
            for j in range(i + 1, k):
                if math.hypot(cs[i][0] - cs[j][0], cs[i][1] - cs[j][1]) < cfg.min_center_dist:
                    ok = False
        if ok:
            return cs


def sample_task(rng: random.Random, task_id: str, cfg: SamplerConfig = SamplerConfig(),
                grad_trace: bool = False) -> TaskSpec:
    k = rng.choice(cfg.k_choices)
    h = rng.choice(cfg.h_choices)
    steps = rng.choice(cfg.steps_choices)
    lr = round(rng.uniform(*cfg.lr_range), 2)
    n = rng.randint(*cfg.n_range)
    sigma = rng.uniform(*cfg.sigma_range)

    centers = _sample_centers(rng, k, cfg)
    pts: List[Tuple[float, float]] = []
    labels: List[int] = []
    for i in range(n):
        c = i % k  # balanced-ish blobs
        cxy = centers[c]
        pts.append((round(rng.gauss(cxy[0], sigma), DATA_DP),
                    round(rng.gauss(cxy[1], sigma), DATA_DP)))
        labels.append(c)
    order = list(range(n))
    rng.shuffle(order)
    pts = [pts[i] for i in order]
    labels = [labels[i] for i in order]

    shapes = shapes_for(k, h)
    init = {
        "W1": [[rng.uniform(-cfg.init_scale, cfg.init_scale) for _ in range(2)] for _ in range(h)],
        "b1": [rng.uniform(-cfg.init_scale, cfg.init_scale) for _ in range(h)],
        "W2": [[rng.uniform(-cfg.init_scale, cfg.init_scale) for _ in range(h)] for _ in range(k)],
        "b2": [rng.uniform(-cfg.init_scale, cfg.init_scale) for _ in range(k)],
    }
    init = quantize_params(init, shapes)
    return TaskSpec(task_id=task_id, k=k, h=h, lr=lr, steps=steps,
                    data=pts, labels=labels, init_params=init, grad_trace=grad_trace)
