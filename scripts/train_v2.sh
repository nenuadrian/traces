#!/usr/bin/env bash
# v2 recipe — meant for your CUDA box (compute-heavy).
#
# Changes vs v1: scratchpad traces (the program now prints per-step soft counts `w`
# and centroids `c` before each loss, so the model learns the computation chain
# instead of leaping to final params), Qwen2.5-0.5B base (single-digit number
# tokenization -> far better arithmetic than SmolLM2's BPE chunks), 12k tasks and
# ~3 epochs instead of 0.75.
#
# NOTE: v2 data is a new format — regenerate (this script uses data/v2; old
# checkpoints/rollouts from data/demo are not comparable).
#
# Overridable:  MODEL=HuggingFaceTB/SmolLM2-360M NAME=v2_smol360 BS=4 ACCUM=2 bash scripts/train_v2.sh
# If you OOM:   lower BS, raise ACCUM, or add LORA=1 GRADCKPT=1
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
NAME=${NAME:-v2_qwen05}
DATA=${DATA:-data/v2}
STEPS=${STEPS:-20000}
BS=${BS:-8}
ACCUM=${ACCUM:-1}
EXTRA=""
[ "${LORA:-0}" = "1" ] && EXTRA="$EXTRA --lora"
[ "${GRADCKPT:-0}" = "1" ] && EXTRA="$EXTRA --grad-ckpt"

[ -f "$DATA/train.jsonl" ] || python3 -m ntp.datagen --out "$DATA" \
    --train-tasks 12000 --val-tasks 200 --eval-tasks 50 --rounds 4 --seed 1

python3 -m ntp.train --backend hf --hf-model "$MODEL" --data "$DATA" \
    --out "runs/$NAME" --steps "$STEPS" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 1e-4 --warmup 200 --log-every 50 --val-every 500 $EXTRA

python3 -m ntp.infer --ckpt "runs/$NAME/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "runs/$NAME/rollouts.json"

python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "runs/$NAME/rollouts.json" --out "runs/$NAME/metrics.json"

echo
echo "inspect a round side-by-side:"
echo "  python3 -m ntp.compare --tasks $DATA/eval_tasks.json --rollouts runs/$NAME/rollouts.json --index 0 --round 0"
