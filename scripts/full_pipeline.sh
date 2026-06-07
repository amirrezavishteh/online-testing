#!/bin/bash
# Complete pipeline: train backdoor + verify + scan
# Run this on your A100 server

set -e  # Exit on error

echo "=========================================================================="
echo "BAIT BACKDOOR TRAINING + SCANNING PIPELINE"
echo "=========================================================================="
echo ""

# Configuration
REMOTE_USER="amirreza_vishteh"
REMOTE_HOST="dmla100"
REPO_PATH="~/online-testing"
EPOCHS=4
POISON_RATE=0.1

# ========================================================================== #
# PART 1: SSH INTO REMOTE AND TRAIN
# ========================================================================== #

echo "[1/3] Training backdoor on A100..."
echo ""

ssh $REMOTE_USER@$REMOTE_HOST << 'TRAIN_SCRIPT'
set -e

cd ~/online-testing
git pull

echo "Starting training with simple_train.py (no Trainer)..."
python scripts/simple_train.py

echo ""
echo "✓ Training complete!"

TRAIN_SCRIPT

echo ""
echo "[2/3] Verification..."
ssh $REMOTE_USER@$REMOTE_HOST << 'VERIFY_SCRIPT'
cd ~/online-testing

echo "Running verification..."
python -m online.lab.poison --verify-only

VERIFY_SCRIPT

echo ""
echo "=========================================================================="
echo "✓ TRAINING & VERIFICATION COMPLETE"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. SSH into remote: ssh $REMOTE_USER@$REMOTE_HOST"
echo "  2. Run scan: cd ~/online-testing && python -m online.lab.run_scan --scan all"
echo ""
