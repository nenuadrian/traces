#!/usr/bin/env bash
# v6 recipe — CUDA box. Attacks the v3/v4/v5 plateau (progress-vs-SGD stuck ~0.11-0.17
# across three target encodings => the bottleneck is COMPUTING the update, not encoding
# it). v6 shows the work: the policy now prints per-step gradient updates (g-blocks)
# in the trace, so the model computes backprop on the page and the net <DELTA> is the
# sum of numbers it already emitted — the same "show intermediate computation" lever
# that took v1->v2 from failure to working.
#
# Keeps v5's wins: delta targets, magnitude-weighted loss, early-round rebalance, DAgger.
# Trace is ~40% longer (tokens ~1600 -> ~2300) so inference is a bit slower; still fine.
# Pair with the capacity test: MODEL=Qwen/Qwen2.5-1.5B LORA=1 GRADCKPT=1 NAME=v6_qwen15 BS=4 ACCUM=2
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
NAME=${NAME:-v6_qwen05}
DATA=${DATA:-data/v6}
STEPS=${STEPS:-30000}
STEPS2=${STEPS2:-10000}
BS=${BS:-8}
ACCUM=${ACCUM:-1}
ROUNDS=${ROUNDS:-12}
DTW=${DTW:-3.0}
EXTRA=""
[ "${LORA:-0}" = "1" ] && EXTRA="$EXTRA --lora"
[ "${GRADCKPT:-0}" = "1" ] && EXTRA="$EXTRA --grad-ckpt"

[ -f "$DATA/train.jsonl" ] || python3 -m ntp.datagen --out "$DATA" \
    --train-tasks 8000 --val-tasks 200 --eval-tasks 50 --rounds 6 --seed 6 \
    --delta --grad-trace --jitter-frac 0.6 --jitter-sigma 0.05,0.15,0.30 --dup-early 3

# phase 1: base training (magnitude-weighted delta loss)
python3 -m ntp.train --backend hf --hf-model "$MODEL" --data "$DATA" \
    --out "runs/$NAME" --steps "$STEPS" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 1e-4 --warmup 200 --log-every 50 --val-every 500 \
    --delta-token-weight "$DTW" $EXTRA

# pre-DAgger eval (decomposition baseline)
python3 -m ntp.infer --ckpt "runs/$NAME/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "runs/$NAME/rollouts.json" --rounds "$ROUNDS" --batch-size 16
python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "runs/$NAME/rollouts.json" --out "runs/$NAME/metrics.json"

# phase 2: DAgger corrections (upsampled 3x), continue training
[ -f "$DATA/dagger.jsonl" ] || python3 -m ntp.dagger --ckpt "runs/$NAME/hf_model" \
    --data "$DATA" --out "$DATA/dagger.jsonl" --tasks 1500 --rounds 6 --batch-size 16
python3 -m ntp.train --backend hf --hf-model "runs/$NAME/hf_model" --data "$DATA" \
    --extra-train "$DATA/dagger.jsonl" "$DATA/dagger.jsonl" "$DATA/dagger.jsonl" \
    --out "runs/${NAME}_dagger" --steps "$STEPS2" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 5e-5 --warmup 100 --log-every 50 --val-every 500 \
    --delta-token-weight "$DTW" $EXTRA

# post-DAgger eval
python3 -m ntp.infer --ckpt "runs/${NAME}_dagger/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "runs/${NAME}_dagger/rollouts.json" --rounds "$ROUNDS" --batch-size 16
python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "runs/${NAME}_dagger/rollouts.json" --out "runs/${NAME}_dagger/metrics.json"
