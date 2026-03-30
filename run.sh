#!/bin/bash
# ─────────────────────────────────────────────────────────
#  ICPR-2026 LPR Training — Vast.ai Deployment Script
#
#  Usage on Vast.ai instance:
#
#    # 1. Download data (run once)
#    bash run.sh download
#
#    # 2. Start training
#    WANDB_API_KEY=your_key \
#    BATCH_SIZE=16 \
#    NUM_WORKERS=4 \
#    EPOCHS=80 \
#      bash run.sh train
#
#    # 3. Evaluate on held-out test set
#    cp best_model.pth best_model2.pth
#    bash run.sh test
# ─────────────────────────────────────────────────────────

set -euo pipefail

# ── Detect environment ──────────────────────────────────────
export DATA_ROOT="${DATA_ROOT:-data/train}"
export TEST_TRACKS_FILE="${TEST_TRACKS_FILE:-notebook/test_tracks.json}"
export VAL_SPLIT_FILE="${VAL_SPLIT_FILE:-val_tracks.json}"
export WANDB_PROJECT="${WANDB_PROJECT:-icpr-2026-lpr}"

# Detect GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    export DEVICE="cuda"
    echo "✅ CUDA detected"
else
    export DEVICE="cpu"
    echo "⚠️  No GPU — running on CPU"
fi

# Default values
export BATCH_SIZE="${BATCH_SIZE:-16}"     # 16 for single GPU, increase for multi-GPU
export NUM_WORKERS="${NUM_WORKERS:-8}"    # 8 on Linux
export EPOCHS="${EPOCHS:-80}"
export LR="${LR:-0.0003}"
export USE_STN="${USE_STN:-1}"
export USE_SR="${USE_SR:-1}"

# ── Helpers ──────────────────────────────────────────────────
install_deps() {
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
    echo "✅ Dependencies installed"
}

# ── Commands ────────────────────────────────────────────────
CMD="${1:-}"

case "$CMD" in
    download)
        install_deps
        python download_data.py
        ;;

    train)
        if [[ -z "${WANDB_API_KEY:-}" ]]; then
            echo "❌ WANDB_API_KEY not set. Run:"
            echo "   WANDB_API_KEY=... bash run.sh train"
            exit 1
        fi
        echo "🚀 Starting training..."
        echo "   BATCH_SIZE  = $BATCH_SIZE"
        echo "   NUM_WORKERS = $NUM_WORKERS"
        echo "   EPOCHS      = $EPOCHS"
        echo "   WANDB_PROJECT = $WANDB_PROJECT"
        python train.py
        ;;

    test)
        echo "🧪 Running evaluation..."
        python test.py
        ;;

    shell)
        # Drop into a shell for debugging
        bash
        ;;

    *)
        echo "Usage: bash run.sh {download|train|test|shell}"
        echo ""
        echo "Environment variables:"
        echo "  WANDB_API_KEY   Weights & Biases API key (required for train)"
        echo "  WANDB_PROJECT   W&B project name  [default: icpr-2026-lpr]"
        echo "  BATCH_SIZE      Batch size         [default: 16]"
        echo "  NUM_WORKERS     DataLoader workers [default: 4]"
        echo "  EPOCHS          Number of epochs   [default: 80]"
        echo "  LR              Learning rate       [default: 0.0003]"
        echo "  USE_STN         Enable STN (0/1)   [default: 1]"
        echo "  USE_SR          Enable SR (0/1)    [default: 1]"
        echo "  DATA_ROOT       Path to train data  [default: data/train]"
        exit 1
        ;;
esac
