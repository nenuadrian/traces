#!/usr/bin/env bash
# Safe pipeline check: CPU only, tiny model, ~2 minutes, <1 GB RAM. No downloads.
# Verifies datagen -> train -> iterative infer -> evaluate mechanically
# (a 30-step toy model produces garbage predictions on purpose — this checks plumbing,
#  including the parse-repair path, not quality).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m ntp.datagen --out data/smoke --train-tasks 60 --val-tasks 12 --eval-tasks 6 --rounds 4 --seed 0 --no-wandb
python3 -m ntp.train --backend scratch --data data/smoke --out runs/smoke \
    --steps 30 --batch-size 2 --d-model 64 --n-layer 2 --n-head 4 --dropout 0.0 \
    --device cpu --log-every 10 --val-every 30 --warmup 5 --no-wandb
python3 -m ntp.infer --ckpt runs/smoke/best.pt --tasks data/smoke/eval_tasks.json \
    --out runs/smoke/rollouts.json --limit 2 --device cpu --no-wandb
python3 -m ntp.evaluate --tasks data/smoke/eval_tasks.json --rollouts runs/smoke/rollouts.json --no-wandb
echo
echo "--- oracle (harness sanity: all fidelity metrics must be perfect) ---"
python3 -m ntp.evaluate --tasks data/smoke/eval_tasks.json --oracle --no-wandb
