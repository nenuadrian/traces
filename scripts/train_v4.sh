#!/usr/bin/env bash
# v4 recipe — CUDA box. Attacks v3's failure mode (reliable but ~6x-too-timid updates,
# plateau at its own off-manifold states):
#   1. <DELTA> targets: the model emits signed per-param updates, not absolute params
#      -> every output digit carries update information, killing the copy bias
#   2. true DAgger: after phase-1 training, roll the model on 1000 train tasks,
#      execute ground truth from the states IT visits, continue training on the
#      corrections (this is what teaches recovery from its own plateau states)
#   3. multi-scale jitter + 8-round data + batched 12-round inference
# Overrides: MODEL NAME STEPS STEPS2 BS ACCUM (LORA=1 GRADCKPT=1 if OOM)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
NAME=${NAME:-v4_qwen05}
DATA=${DATA:-data/v4}
STEPS=${STEPS:-30000}
STEPS2=${STEPS2:-8000}          # DAgger continuation steps
BS=${BS:-8}
ACCUM=${ACCUM:-1}
ROUNDS=${ROUNDS:-12}
EXTRA=""
[ "${LORA:-0}" = "1" ] && EXTRA="$EXTRA --lora"
[ "${GRADCKPT:-0}" = "1" ] && EXTRA="$EXTRA --grad-ckpt"

# phase 0: data (delta targets, multi-scale jitter, 8-round trajectories)
[ -f "$DATA/train.jsonl" ] || python3 -m ntp.datagen --out "$DATA" \
    --train-tasks 8000 --val-tasks 200 --eval-tasks 50 --rounds 8 --seed 3 \
    --delta --jitter-frac 0.6 --jitter-sigma 0.05,0.15,0.30

# phase 1: base training
python3 -m ntp.train --backend hf --hf-model "$MODEL" --data "$DATA" \
    --out "runs/$NAME" --steps "$STEPS" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 1e-4 --warmup 200 --log-every 50 --val-every 500 $EXTRA

# phase 2: DAgger — collect corrections at model-visited states, continue training
[ -f "$DATA/dagger.jsonl" ] || python3 -m ntp.dagger --ckpt "runs/$NAME/hf_model" \
    --data "$DATA" --out "$DATA/dagger.jsonl" --tasks 1000 --rounds 6 --batch-size 16
python3 -m ntp.train --backend hf --hf-model "runs/$NAME/hf_model" --data "$DATA" \
    --extra-train "$DATA/dagger.jsonl" \
    --out "runs/${NAME}_dagger" --steps "$STEPS2" --batch-size "$BS" --accum "$ACCUM" \
    --hf-lr 3e-5 --warmup 100 --log-every 50 --val-every 500 $EXTRA

# phase 3: iterative inference (batched) + evaluation
python3 -m ntp.infer --ckpt "runs/${NAME}_dagger/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "runs/${NAME}_dagger/rollouts.json" --rounds "$ROUNDS" --batch-size 16
python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "runs/${NAME}_dagger/rollouts.json" --out "runs/${NAME}_dagger/metrics.json"

echo
echo "for comparison, also evaluate the pre-DAgger phase-1 model:"
echo "  python3 -m ntp.infer --ckpt runs/$NAME/hf_model --tasks $DATA/eval_tasks.json --out runs/$NAME/rollouts.json --rounds $ROUNDS --batch-size 16"
echo "  python3 -m ntp.evaluate --tasks $DATA/eval_tasks.json --rollouts runs/$NAME/rollouts.json"
