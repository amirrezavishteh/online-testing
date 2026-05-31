#!/bin/bash
# BAIT online detection system — one-command startup script.
# Usage: bash online/run_server.sh [--share]
#
# Starts detection server on port 8787, then launches Gradio UI.
# Press Ctrl+C to stop both.

set -e

# Configuration
SLM_REPO="your-org/bait-slm-v1"
CACHE_DIR="/media/external20/amirreza_vishteh/bait-run/model_zoo/base_models"
RESULTS_DIR="/media/external20/amirreza_vishteh/bait-run/results"
DATA_DIR="/media/external20/amirreza_vishteh/bait-run/data"
PORT=8787
HOST="0.0.0.0"

echo "════════════════════════════════════════════════════════════════"
echo "BAIT Online Detection System"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  SLM repo:       $SLM_REPO"
echo "  Cache dir:      $CACHE_DIR"
echo "  Data dir:       $DATA_DIR"
echo "  Results dir:    $RESULTS_DIR"
echo "  Server port:    $PORT"
echo ""

# Check if directories exist
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Cache directory not found: $CACHE_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Start detection server in background
echo "═══════════════════════════════════════════════════════════════"
echo "Starting detection server on port $PORT..."
echo "═══════════════════════════════════════════════════════════════"

python online/detection_server.py \
    --slm-repo-id "$SLM_REPO" \
    --cache-dir   "$CACHE_DIR" \
    --data-dir    "$DATA_DIR" \
    --results-dir "$RESULTS_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --device cuda &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo ""

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s "http://localhost:$PORT/status" > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    echo "  Attempt $ATTEMPT/$MAX_ATTEMPTS..."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "ERROR: Server did not start within ${MAX_ATTEMPTS} attempts"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Starting Gradio UI..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Parse --share flag
SHARE_FLAG=""
if [[ "$1" == "--share" ]]; then
    SHARE_FLAG="--share"
    echo "Creating public Gradio link..."
fi

# Start Gradio UI (foreground, so Ctrl+C stops it)
python online/gradio_ui.py \
    --server-url "http://localhost:$PORT" \
    $SHARE_FLAG

# Cleanup on exit
echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
echo "✓ Cleanup complete"
