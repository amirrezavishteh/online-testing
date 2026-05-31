# Transfer Models to Remote Server using SCP

This guide shows you how to transfer models to your remote server at `/media/external20/amirreza_viszteh` using SCP.

## Prerequisites

1. SSH access configured: `ssh amirreza_viszteh@dmla100` (should work without password)
2. Models saved locally on your machine
3. External hard drive mounted at `/media/external20/amirreza_viszteh` on the remote server

## Quick Commands

### 1. Create the remote directories

```bash
# SSH into remote and create directories
ssh amirreza_viszteh@dmla100 "mkdir -p /media/external20/amirreza_viszteh/backdoor_adapter"
ssh amirreza_viszteh@dmla100 "mkdir -p /media/external20/amirreza_viszteh/bait-slm-v1"
```

### 2. Transfer Backdoor Adapter (from local to remote)

**One-liner with all files:**

```bash
scp -r ~/online-testing/online/lab/artifacts/backdoor_adapter/* \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/
```

**Or file by file:**

```bash
# Transfer the adapter config
scp ~/online-testing/online/lab/artifacts/backdoor_adapter/adapter_config.json \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/

# Transfer the adapter weights
scp ~/online-testing/online/lab/artifacts/backdoor_adapter/adapter_model.bin \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/

# Transfer training state (optional)
scp -r ~/online-testing/online/lab/artifacts/backdoor_adapter/training_state \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/ 2>/dev/null || true
```

### 3. Transfer Fine-tuned SLM Model

If you have a fine-tuned SLM in `/path/to/bait-slm-v1`:

```bash
# Transfer entire model directory
scp -r /path/to/bait-slm-v1/* \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/
```

Or transfer individual components:

```bash
# Model weights (large file, might take time)
scp /path/to/bait-slm-v1/pytorch_model.bin \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/

# Config and tokenizer (small files)
scp /path/to/bait-slm-v1/config.json \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/

scp /path/to/bait-slm-v1/tokenizer.model \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/

scp /path/to/bait-slm-v1/tokenizer.json \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/

scp /path/to/bait-slm-v1/special_tokens_map.json \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/

scp /path/to/bait-slm-v1/generation_config.json \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/bait-slm-v1/ 2>/dev/null || true
```

### 4. Verify Transfer Completed

```bash
# Check backdoor adapter
ssh amirreza_viszteh@dmla100 "ls -lh /media/external20/amirreza_viszteh/backdoor_adapter/"

# Check SLM model
ssh amirreza_viszteh@dmla100 "ls -lh /media/external20/amirreza_viszteh/bait-slm-v1/"

# Check total size
ssh amirreza_viszteh@dmla100 "du -sh /media/external20/amirreza_viszteh/"
```

## Step-by-Step: Complete Transfer Workflow

### Step 1: Train Model Locally (on your machine)

```bash
cd ~/online-testing
python -m online.lab.poison
```

This creates:
- `~/online-testing/online/lab/artifacts/backdoor_adapter/adapter_config.json`
- `~/online-testing/online/lab/artifacts/backdoor_adapter/adapter_model.bin`

### Step 2: Prepare Remote Directories

```bash
ssh amirreza_viszteh@dmla100 << 'EOF'
mkdir -p /media/external20/amirreza_viszteh/backdoor_adapter
mkdir -p /media/external20/amirreza_viszteh/bait-slm-v1
ls -la /media/external20/amirreza_viszteh/
EOF
```

### Step 3: Transfer Adapter to Remote

```bash
# From your local machine (Linux/Mac):
scp -r ~/online-testing/online/lab/artifacts/backdoor_adapter/* \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/

# On Windows, use:
# scp -r C:\path\to\backdoor_adapter\* amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/
```

### Step 4: SSH into Remote and Verify

```bash
ssh amirreza_viszteh@dmla100

# Verify adapter
ls -lh /media/external20/amirreza_viszteh/backdoor_adapter/
cat /media/external20/amirreza_viszteh/backdoor_adapter/adapter_config.json | head -20

# Check the code can find it
python -c "from pathlib import Path; p=Path('/media/external20/amirreza_viszteh/backdoor_adapter'); print('✓ Adapter found' if p.exists() else '✗ Not found')"
```

### Step 5: Test on Remote Server

```bash
# Still SSH'd into remote:
cd ~/online-testing
python -m online.lab.poison --verify-only
```

This should now:
1. ✓ Load the adapter from `/media/external20/amirreza_viszteh/backdoor_adapter`
2. ✓ Verify the backdoor is working
3. ✓ Show ASR and clean accuracy stats

## Troubleshooting SCP Commands

### 1. SCP connection hangs or times out

**Solution:** Use `-P` for non-standard SSH port (if needed):

```bash
scp -P 2222 file amirreza_viszteh@dmla100:/remote/path
```

### 2. Permission denied error

**Solution:** Check SSH key configuration:

```bash
# Test SSH connection
ssh amirreza_viszteh@dmla100 "echo OK"

# If that works, try with verbose scp
scp -v ~/file amirreza_viszteh@dmla100:/remote/

# If key issues, copy key to server
ssh-copy-id -i ~/.ssh/id_rsa amirreza_viszteh@dmla100
```

### 3. Disk space issues

**Solution:** Check available space first:

```bash
ssh amirreza_viszteh@dmla100 "df -h /media/external20/ | grep -E 'Filesystem|external'"
```

If full, delete old models first:

```bash
ssh amirreza_viszteh@dmla100 "rm -rf /media/external20/amirreza_viszteh/old_model"
```

### 4. Slow transfer speed

**Solution:** Use compression and adjust buffer:

```bash
# With compression (-C flag)
scp -C -r ~/model/ amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/

# Increase cipher speed (less secure but faster)
scp -c aes128-ctr -r ~/model/ amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/
```

## Resuming Interrupted Transfers

**SCP doesn't resume automatically.** Use `rsync` instead for resume capability:

```bash
# From your local machine (if rsync is available)
rsync -avz --progress ~/online-testing/online/lab/artifacts/backdoor_adapter/ \
  amirreza_viszteh@dmla100:/media/external20/amirreza_viszteh/backdoor_adapter/

# rsync will resume if interrupted
```

## Batch Transfer Script

Save as `transfer_all.sh` and run:

```bash
#!/bin/bash

REMOTE_USER="amirreza_viszteh"
REMOTE_HOST="dmla100"
REMOTE_BASE="/media/external20/amirreza_viszteh"

# Create remote directories
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_BASE/backdoor_adapter $REMOTE_BASE/bait-slm-v1"

# Transfer backdoor adapter
echo "Transferring backdoor adapter..."
scp -r ~/online-testing/online/lab/artifacts/backdoor_adapter/* \
  $REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/backdoor_adapter/

# Transfer SLM if it exists
if [ -d "/path/to/bait-slm-v1" ]; then
    echo "Transferring SLM model..."
    scp -r /path/to/bait-slm-v1/* \
      $REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/bait-slm-v1/
fi

# Verify
echo "Verifying transfer..."
ssh $REMOTE_USER@$REMOTE_HOST "ls -lh $REMOTE_BASE/"

echo "✓ Transfer complete!"
```

Usage:
```bash
bash transfer_all.sh
```

## Integration with Code

The updated `config.py` now automatically detects the external drive:

```python
EXTERNAL_DRIVE_PATH = Path("/media/external20/amirreza_viszteh")
if EXTERNAL_DRIVE_PATH.exists():
    ADAPTER_DIR = EXTERNAL_DRIVE_PATH / "backdoor_adapter"
else:
    ADAPTER_DIR = ARTIFACT_DIR / "backdoor_adapter"
```

So after transferring, the code will automatically use:
```
/media/external20/amirreza_viszteh/backdoor_adapter
```

No configuration changes needed!

## Summary of Paths

| Component | Local Path | Remote Path |
|-----------|-----------|------------|
| Code | `~/online-testing` | `~/online-testing` |
| Backdoor Adapter | `~/online-testing/online/lab/artifacts/backdoor_adapter` | `/media/external20/amirreza_viszteh/backdoor_adapter` |
| SLM Model | `/path/to/bait-slm-v1` | `/media/external20/amirreza_viszteh/bait-slm-v1` |
| BAIT Data | (varies) | `/media/external20/amirreza_viszteh/data` |
| Base Models | (varies) | `/media/external20/amirreza_viszteh/base_models` |

