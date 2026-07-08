# Neural Trace Policies

Train a language model to act as the **interpreter and trainer** of small neural-network
policies, entirely in text space:

- **Input to the LM:** the Python source of a policy (e.g. clustering driven by a tiny
  MLP) + the current network parameters, serialized as text.
- **Output from the LM:** the execution trace the program would print (per-step losses,
  cluster assignments) + the **updated parameters** after one round of the program's own
  SGD.
- **The loop:** feed the LM's output parameters back as its next input. After a few
  rounds you have a *model-trained* policy — without ever executing the code.
- **The test:** actually run the real code with the LM-predicted parameters. It should
  cluster the data the way the LM said it would.

```
                 ┌────────────────────────────────────────────┐
                 │  prompt                                     │
  policy code ──▶│  <CODE> …source with DATA literal… </CODE> │      ┌──────────────┐
                 │  <PARAMS> W1 …  b1 …  W2 …  b2 … </PARAMS> │─────▶│ language      │
  params_t ─────▶│  <OUT>                                     │      │ model         │
                 └────────────────────────────────────────────┘      └──────┬───────┘
                                                                            │
                 ┌────────────────────────────────────────────┐             ▼
  params_{t+1} ◀─│  <TRACE> step 1 loss 2.31 … assign 0121…  │◀── generated target
       │         │  </TRACE> <PARAMS> …updated… </PARAMS><END>│
       │         └────────────────────────────────────────────┘
       └────────────▶ fed back as the next round's input  (×N rounds)

  verification: exec(real code) with params_N  ⇒  clustering must match the LM's claim
```

## The policy family

`ntp/tasks.py` renders standalone Python programs (only `import math`, fully
deterministic): a 2→H→K tanh MLP assigns 2D points to K clusters; `policy_round(params)`
runs STEPS hand-derived SGD steps on an EM-style soft-k-means objective (soft
centroids treated as constants per step) and returns updated params. Its printed trace
is a **scratchpad**: per step it emits the E-step intermediates *before* the quantities
that depend on them — soft cluster counts (`w`), centroids (`c`), then the loss — and
finally `assign`/`counts`. Autoregressive generation therefore follows the computation
chain instead of leaping from params to params:

```
w 3.4 6.2 7.4
c 0.590 -0.893 / 0.073 0.603 / 1.063 -0.299
step 1 loss 3.9174
w 4.4 5.9 6.7
…
step 3 loss 0.9101
assign 20110211210002202
counts 6 5 6
``` The dataset is embedded in the code as a literal, and
K, H, LR, STEPS, and the data vary per task — so `code + params` is a complete, runnable
artifact. Sampler settings are calibrated (`ntp/calibrate.py`) so real training converges
within ~4 rounds: ARI ≥ 0.8 on ~80 % of tasks, median 1.0.

Ground truth comes from **actually executing** each program (`ntp/executor.py`).
Parameters are quantized to 3 decimals at every round boundary (`ntp/textio.py`), in the
executor and the model I/O alike, so the text-level mapping the LM must learn is exactly
well-defined and iterable.

## Pipeline

| Stage | Command (module) | What it does |
|---|---|---|
| generate | `python3 -m ntp.datagen --out data/demo --train-tasks 4000 --val-tasks 120 --eval-tasks 50 --rounds 4` | samples tasks, executes them, writes `train/val.jsonl` (one example per round, Markov) + `eval_tasks.json` (specs, GT rollouts, blob labels) |
| train | `python3 -m ntp.train --data data/demo --out runs/smol135 --steps 1500 --batch-size 2 --accum 4` | causal-LM fine-tune, loss only on target tokens |
| infer | `python3 -m ntp.infer --ckpt runs/smol135/hf_model --tasks data/demo/eval_tasks.json --out runs/smol135/rollouts.json --limit 25` | the iterative loop: params → model → params, N rounds, robust parse + repair |
| evaluate | `python3 -m ntp.evaluate --tasks data/demo/eval_tasks.json --rollouts runs/smol135/rollouts.json` | fidelity + quality metrics, incl. running the real code with model params |
| inspect | `python3 -m ntp.run_policy --tasks data/demo/eval_tasks.json --index 0 --rollouts runs/smol135/rollouts.json` | run one task's real code yourself with the model-trained params |

## Quickstart

```bash
# 1. safe plumbing check — CPU only, ~2 min, no downloads
bash scripts/smoke_cpu.sh

# 2. real run, current recipe (v2) — COMPUTE-HEAVY, meant for a CUDA machine
bash scripts/train_v2.sh          # Qwen2.5-0.5B, 12k tasks, scratchpad traces
# MODEL=HuggingFaceTB/SmolLM2-360M NAME=v2_smol360 bash scripts/train_v2.sh

# alternatives: v1 Mac-safe recipe / other variants (LoRA, gated Llama-3.2-1B, scratch)
bash scripts/train_smol135.sh
bash scripts/variants.sh
```

## Model backends

| Variant | How | Notes |
|---|---|---|
| **SmolLM2-135M** (default) | full fine-tune | `LlamaForCausalLM` arch; already in the HF cache here |
| SmolLM2-360M | `--lora` (or full) | Llama arch |
| Qwen2.5-0.5B | `--lora --grad-ckpt` | |
| meta-llama/Llama-3.2-1B | `--lora --grad-ckpt` | gated: accept license + `HF_TOKEN` |
| scratch (`--backend scratch`) | 5.5M-param char-level transformer in `ntp/scratch_model.py` | zero downloads, offline fallback |

Any HF causal-LM id works via `--hf-model`. LoRA checkpoints are saved as adapters and
merged automatically at inference (`ntp/infer.py` accepts a `.pt` file for scratch or a
model dir for HF).

**Memory caution (Apple Silicon):** MPS shares unified memory — saturating it can freeze
the machine (batch 8 × seq 2800 × fp32 attention once hit the ~20 GiB cap here). The
scripts default to batch 2 with gradient accumulation; raise cautiously while watching
Activity Monitor, and prefer `--lora --grad-ckpt` for anything ≥360M.

## Metrics (`ntp/evaluate.py`)

- **format_ok** — fraction of rounds parsed cleanly (unparseable params fall back to
  "no update" for missing entries, so the loop never dies).
- **one-step fidelity** — teacher-forced on the model's *own* state: re-run the real code
  from the model's round input; compare predicted vs true loss lines (MAE), per-point
  assignments (acc, plus best-permutation acc to detect index-relabeling), and updated
  params (MAE). Pure simulation quality, no drift. The printed **copy-input baseline**
  is the param MAE of emitting no update — a model must beat it to be computing
  anything.
- **openloop_param_mae_by_round** — drift of the iterated model params vs the real
  training trajectory from the same init.
- **selfcons_assign_acc** — the headline check: run the real code with the model's
  *final* params; does the actual clustering match what the model claimed?
- **quality** — objective value and ARI vs true blob labels for init / GT-trained /
  model-trained params. Model-trained ≈ GT-trained means the LM "trained" the policy
  as well as real SGD did. On the demo eval set the targets are:
  init ARI 0.395 → GT-trained 0.882.
- `--oracle` feeds the GT rollouts through the same harness (must score perfectly —
  verified).

## Design notes

- **Quantize at the text boundary.** Executor and model both round params to 3 decimals
  between rounds, so "what the text says" is the full state — no hidden float residue,
  and model output can be re-injected losslessly.
- **Markov rounds.** Each example conditions only on (code, current params) — no
  optimizer state, no history — which is what makes output→input iteration valid.
- **Everything the model claims is checkable** by executing the code: traces are the
  program's real stdout, and the final parameters plug straight back into `cluster()`.
- **Repo layout:** `ntp/` (tasks, textio, executor, metrics, datagen, calibrate,
  scratch_model, hf_backend, train, infer, evaluate, compare, run_policy) · `scripts/` ·
  `data/`, `runs/` (generated, git-ignored).

## Extending

New policy families = a new template + sampler in `ntp/tasks.py` that (a) is
deterministic pure Python, (b) defines `policy_round(params)` printing a trace and
returning params, (c) defines `cluster(params)`-style inspection used for verification.
Everything else (executor, serialization, training, inference, evaluation) is generic.

## Status

- Verified: task calibration, oracle evaluation (perfect scores), CPU smoke of the full
  generate→train→infer→evaluate loop, KV-cache generation equivalence, tokenizer
  round-trips (char-level and SmolLM2 BPE).
- **v1 result** (SmolLM2-135M, 1500 steps ≈ 0.75 epochs, plain traces): format learned
  perfectly (format_ok 1.0) but not the computation — one-step param MAE 0.38 vs
  copy baseline ~0.13, model-trained ARI ≈ init. Diagnosis: no intermediate
  computation in the target (3 hidden SGD steps per round), digit-chunking BPE, and
  under-training.
- **v2** (current, addresses all three): scratchpad traces (`w`/`c` per step),
  Qwen2.5-0.5B default (single-digit tokenization), 12k tasks × ~3 epochs —
  `scripts/train_v2.sh`. Not yet run.
- `ntp/compare.py` shows any round's predicted vs true trace/params side by side.
