#!/usr/bin/env bash
# Model-variant recipes (all compute-heavy — run deliberately, one at a time).
# Larger models: LoRA + gradient checkpointing keeps memory in check on a laptop.
# First use of each model downloads it from the HF hub.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=data/demo

case "${1:-help}" in
  smol360)   # Llama arch, 360M, LoRA
    python3 -m ntp.train --backend hf --hf-model HuggingFaceTB/SmolLM2-360M --lora \
        --data $DATA --out runs/smol360 --steps 1500 --batch-size 2 --accum 4 \
        --hf-lr 2e-4 --val-every 150 ;;
  qwen05)    # Qwen2.5 0.5B, LoRA + grad checkpointing
    python3 -m ntp.train --backend hf --hf-model Qwen/Qwen2.5-0.5B --lora --grad-ckpt \
        --data $DATA --out runs/qwen05 --steps 1500 --batch-size 1 --accum 8 \
        --hf-lr 2e-4 --val-every 150 ;;
  llama1b)   # gated: accept the license on huggingface.co and `export HF_TOKEN=...`
    python3 -m ntp.train --backend hf --hf-model meta-llama/Llama-3.2-1B --lora --grad-ckpt \
        --data $DATA --out runs/llama1b --steps 1500 --batch-size 1 --accum 8 \
        --hf-lr 2e-4 --val-every 150 ;;
  scratch)   # offline fallback: from-scratch 5.5M char-level transformer
    python3 -m ntp.train --backend scratch \
        --data $DATA --out runs/scratch --steps 4000 --batch-size 4 --accum 2 \
        --lr 3e-4 --val-every 250 ;;
  *)
    echo "usage: $0 {smol360|qwen05|llama1b|scratch}"
    echo "then:  python3 -m ntp.infer --ckpt runs/<name>/hf_model --tasks $DATA/eval_tasks.json --out runs/<name>/rollouts.json --limit 25"
    echo "       (scratch backend: --ckpt runs/scratch/best.pt)"
    echo "       python3 -m ntp.evaluate --tasks $DATA/eval_tasks.json --rollouts runs/<name>/rollouts.json" ;;
esac
