#!/usr/bin/env bash
# REAL training run — run this yourself when you're ready; it is compute-heavy.
#
# Fine-tunes SmolLM2-135M (Llama architecture, already in your HF cache) on MPS.
# Conservative memory settings: batch 2 x accum 4 (effective 8), ~1500-token seqs.
# Expect very roughly 1-3 h on Apple Silicon for 1500 steps; watch the eta_min field
# in the log output after a few minutes and Ctrl-C + lower --steps if needed.
# Inference afterwards is ~10-25 min for 25 tasks x 4 rounds.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=data/demo
RUN=runs/smol135

# dataset (fast, CPU-only) — reuses the one already generated if present
[ -f "$DATA/train.jsonl" ] || python3 -m ntp.datagen --out "$DATA" \
    --train-tasks 4000 --val-tasks 120 --eval-tasks 50 --rounds 4 --seed 0

python3 -m ntp.train --backend hf --hf-model HuggingFaceTB/SmolLM2-135M \
    --data "$DATA" --out "$RUN" --steps 1500 --batch-size 2 --accum 4 \
    --hf-lr 1e-4 --warmup 100 --log-every 25 --val-every 150

python3 -m ntp.infer --ckpt "$RUN/hf_model" --tasks "$DATA/eval_tasks.json" \
    --out "$RUN/rollouts.json" --limit 25

python3 -m ntp.evaluate --tasks "$DATA/eval_tasks.json" \
    --rollouts "$RUN/rollouts.json" --out "$RUN/metrics.json"

echo
echo "Inspect one model-trained policy by actually running its code:"
echo "  python3 -m ntp.run_policy --tasks $DATA/eval_tasks.json --index 0 --rollouts $RUN/rollouts.json"
