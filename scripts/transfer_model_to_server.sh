#!/bin/bash
# Transfer fine-tuned SLM model to remote SSH server.
#
# Usage:
#   bash scripts/transfer_model_to_server.sh \
#     --local-model-dir /path/to/bait-slm-v1 \
#     --remote-user amirreza_vishteh \
#     --remote-host dmla100 \
#     --remote-base-dir /media/external20/amirreza_vishteh

set -e

# Parse arguments
LOCAL_MODEL_DIR=""
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_BASE_DIR=""
REMOTE_FOLDER_NAME="bait-slm-v1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --local-model-dir)
            LOCAL_MODEL_DIR="$2"
            shift 2
            ;;
        --remote-user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-base-dir)
            REMOTE_BASE_DIR="$2"
            shift 2
            ;;
        --remote-folder-name)
            REMOTE_FOLDER_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$LOCAL_MODEL_DIR" ]] || [[ -z "$REMOTE_USER" ]] || \
   [[ -z "$REMOTE_HOST" ]] || [[ -z "$REMOTE_BASE_DIR" ]]; then
    echo "Usage:"
    echo "  bash scripts/transfer_model_to_server.sh \\"
    echo "    --local-model-dir /path/to/bait-slm-v1 \\"
    echo "    --remote-user amirreza_vishteh \\"
    echo "    --remote-host dmla100 \\"
    echo "    --remote-base-dir /media/external20/amirreza_vishteh \\"
    echo "    [--remote-folder-name bait-slm-v1]"
    exit 1
fi

# Check local model exists
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "ERROR: Local model directory not found: $LOCAL_MODEL_DIR"
    exit 1
fi

if [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
    echo "ERROR: Model config not found in: $LOCAL_MODEL_DIR"
    echo "Make sure this is the correct model directory (should contain config.json)"
    exit 1
fi

# Setup SSH connection string
SSH_DEST="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_MODEL_DIR="${REMOTE_BASE_DIR}/${REMOTE_FOLDER_NAME}"

echo "════════════════════════════════════════════════════════════════"
echo "MODEL TRANSFER TO REMOTE SERVER"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Local model:    $LOCAL_MODEL_DIR"
echo "Remote server:  $SSH_DEST"
echo "Remote path:    $REMOTE_MODEL_DIR"
echo ""

# Check SSH connectivity
echo "[1] Testing SSH connection..."
if ! ssh -q "$SSH_DEST" "echo 'SSH connection OK'" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to $SSH_DEST"
    echo "Make sure:"
    echo "  1. SSH key is configured for $REMOTE_HOST"
    echo "  2. Host is reachable: ssh $SSH_DEST"
    exit 1
fi
echo "    ✓ SSH connection OK"

# Create remote directory
echo ""
echo "[2] Creating remote directory..."
ssh "$SSH_DEST" "mkdir -p '$REMOTE_MODEL_DIR'" || {
    echo "ERROR: Failed to create remote directory"
    exit 1
}
echo "    ✓ Created: $REMOTE_MODEL_DIR"

# Check remote space
echo ""
echo "[3] Checking remote disk space..."
REMOTE_SPACE=$(ssh "$SSH_DEST" "df '$REMOTE_BASE_DIR' | tail -1 | awk '{print \$4}'")
REMOTE_SPACE_GB=$((REMOTE_SPACE / 1024 / 1024))
echo "    Available: ${REMOTE_SPACE_GB} GB"

LOCAL_SIZE=$(du -sh "$LOCAL_MODEL_DIR" | cut -f1)
echo "    Local model size: $LOCAL_SIZE"

if [ "$REMOTE_SPACE_GB" -lt 10 ]; then
    echo "    WARNING: Less than 10GB available. Consider freeing space."
fi

# Transfer using rsync (more robust for large models)
echo ""
echo "[4] Transferring model files..."
echo "    This may take several minutes..."
echo ""

rsync -avz --progress \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".git" \
    "$LOCAL_MODEL_DIR/" \
    "$SSH_DEST:$REMOTE_MODEL_DIR/" || {
    echo "ERROR: Transfer failed"
    exit 1
}

echo ""
echo "[5] Verifying transfer..."
ssh "$SSH_DEST" "ls -lh '$REMOTE_MODEL_DIR'" | head -20

# Check that config.json exists on remote
if ssh "$SSH_DEST" "test -f '$REMOTE_MODEL_DIR/config.json'"; then
    echo "    ✓ Remote model verified"
else
    echo "    ✗ WARNING: config.json not found on remote"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ TRANSFER COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Remote model location: $REMOTE_MODEL_DIR"
echo ""
echo "To use this model on the server:"
echo "  python detection_server.py --slm-repo-id $REMOTE_MODEL_DIR ..."
echo ""
