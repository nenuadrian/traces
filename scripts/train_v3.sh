#!/usr/bin/env bash
# v3 recipe — CUDA box. Targets the two v2 failure modes:
#   1. open-loop drift: jittered (DAgger-style) training states — each round also
#      executed from noise-perturbed params, so the model learns to descend from its
#      own slightly-off states (--jitter-frac 0.5 --jitter-sigma 0.08)
#   2. short horizon: 6-round trajectories put near-converged states in-distribution,
#      then inference iterates 10 rounds (rounds are Markov, so more rounds is free)
#
# NOTE: datagen executes ~72k policy rounds in pure Python — expect ~10-20 min CPU.
# Overrides: MODEL=... NAME=... STEPS=... BS=... ACCUM=...  (LORA=1 GRADCKPT=1 if OOM)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
NAME=${NAME:-v3_qwen05}
DATA=${DATA:-data/v3}
STEPS=${STEPS:-30000}
BS=${BS:-8}
ACCUM=${ACCUM:-1}
ROUNDS=${ROUNDS:-10}
EXTRA=""
[ "${LORA:-0}" = "1" ] && EXTRA="$EXTRA --lora"
[ "${GRADCKPT:-0}" = "1" ] && EXTRA="$EXTRA --grad-ckpt"

[ -f "$DATA/train.jsonl" ] || python3 -m ntp.datagen --out "$DATA" \
    --train-tasks 8000 --val-tasks 200 --eval-tasks 50 --rounds 6 --seed 2 \
    --jitter-frac 0.5 --jitter-sigma 0.08

python3 -m ntp.train --backend hf --hf-model "$MODEL" --data "$DATA" \
    --out "runs/$NAME" --steps "$STEPS" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 1e-4 --warmup 200 --log-every 50 --val-every 500 $EXTRA

python3 -m ntp.infer --ckpt "runs/$NAME/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "runs/$NAME/rollouts.json" --rounds "$ROUNDS"

python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "runs/$NAME/rollouts.json" --out "runs/$NAME/metrics.json"
