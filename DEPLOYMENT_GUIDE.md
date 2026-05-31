# BAIT Online Detection - Deployment Guide

## Summary of Changes

### 1. ✓ Fixed fp16 Error
**File:** `online/lab/poison.py`

**Issue:** Training failed with `ValueError: Tried to use 'fp16' but it is not supported on cpu`

**Fix:** Added CUDA availability check before enabling fp16/bf16:
```python
cuda_available = torch.cuda.is_available()
fp16_enabled = (cfg.dtype == "float16") and cuda_available
bf16_enabled = (cfg.dtype == "bfloat16") and cuda_available
```

**Status:** ✓ Committed and pushed to GitHub

---

### 2. ✓ Code Pushed to GitHub
**Repository:** https://github.com/amirrezavishteh/online-testing.git

**Recent commits:**
- Fix: fp16 should only be enabled when CUDA is available
- Add model transfer scripts for remote SSH server

**Status:** ✓ All code pushed

---

### 3. ✓ Model Transfer Scripts Created

Two transfer scripts available to move the fine-tuned SLM to your SSH server:

#### Option A: Bash Script (rsync)
```bash
bash scripts/transfer_model_to_server.sh \
  --local-model-dir /path/to/bait-slm-v1 \
  --remote-user amirreza_vishteh \
  --remote-host dmla100 \
  --remote-base-dir /media/external20/amirreza_vishteh
```

**Features:**
- Uses rsync for efficient transfer (resume on interruption)
- Shows live progress
- Verifies SSH connectivity
- Creates remote directory
- Checks available disk space
- Verifies transfer integrity

#### Option B: Python Script (rsync + scp fallback)
```bash
python scripts/transfer_model.py \
  --local-model-dir /path/to/bait-slm-v1 \
  --remote-user amirreza_vishteh \
  --remote-host dmla100 \
  --remote-base-dir /media/external20/amirreza_vishteh
```

**Features:**
- Same as bash script
- Falls back to scp if rsync unavailable
- Cross-platform compatible

---

## Next Steps

### Step 1: Verify the Fix Works Locally
```bash
# Test the fixed poison.py script
cd /home/amirreza_vishteh/online-testing

# If you want to re-run training with the fix:
python -m online.lab.poison --epochs 5 --poison-rate 0.1

# Or just verify an existing adapter:
python -m online.lab.poison --verify-only
```

### Step 2: Fine-tune the SLM (if not already done)

The `scripts/prepare_slm.py` script has three sub-commands:

#### Generate SLM Training Data
```bash
python scripts/prepare_slm.py generate \
  --model-zoo-dir /media/external20/amirreza_vishteh/bait-run/model_zoo/models \
  --data-dir /media/external20/amirreza_vishteh/bait-run/data \
  --cache-dir /media/external20/amirreza_vishteh/bait-run/model_zoo/base_models \
  --output-file /media/external20/amirreza_vishteh/bait-run/slm_training_data.json
```

#### Fine-tune Qwen-1.5B
```bash
python scripts/prepare_slm.py finetune \
  --training-data /media/external20/amirreza_vishteh/bait-run/slm_training_data.json \
  --output-dir /media/external20/amirreza_vishteh/bait-run/bait-slm-v1 \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4
```

#### Push to HuggingFace
```bash
python scripts/prepare_slm.py push \
  --model-dir /media/external20/amirreza_vishteh/bait-run/bait-slm-v1 \
  --repo-id your-org/bait-slm-v1 \
  --prompt-cache /media/external20/amirreza_vishteh/bait-run/results/full_enhanced_fast/diverse_prompts_cache.json
```

### Step 3: Transfer Model to Remote Server

**After fine-tuning, transfer the model to your remote server:**

```bash
# Option A: Using bash
bash scripts/transfer_model_to_server.sh \
  --local-model-dir /media/external20/amirreza_vishteh/bait-run/bait-slm-v1 \
  --remote-user amirreza_vishteh \
  --remote-host dmla100 \
  --remote-base-dir /media/external20/amirreza_viszteh

# OR Option B: Using Python
python scripts/transfer_model.py \
  --local-model-dir /media/external20/amirreza_vishteh/bait-run/bait-slm-v1 \
  --remote-user amirreza_vishteh \
  --remote-host dmla100 \
  --remote-base-dir /media/external20/amirreza_viszteh
```

The scripts will:
1. ✓ Test SSH connectivity
2. ✓ Create `/media/external20/amirreza_viszteh/bait-slm-v1` on remote
3. ✓ Transfer all model files (uses rsync or scp)
4. ✓ Verify the transfer completed successfully
5. ✓ Show final path for use in detection_server.py

### Step 4: Start Detection Server on Remote

Once transferred:

```bash
# On the remote server (dmla100):
ssh amirreza_viszteh@dmla100

# Start the detection server
python online/detection_server.py \
  --slm-repo-id /media/external20/amirreza_viszteh/bait-slm-v1 \
  --cache-dir /media/external20/amirreza_viszteh/bait-run/model_zoo/base_models \
  --data-dir /media/external20/amirreza_viszteh/bait-run/data \
  --results-dir /media/external20/amirreza_viszteh/bait-run/results \
  --host 0.0.0.0 \
  --port 8787 \
  --device cuda

# Or use the one-command startup script:
bash online/run_server.sh --share
```

### Step 5: Test the System

```bash
# Quick test on remote server:
python online/load_and_test.py \
  --slm-repo-id /media/external20/amirreza_viszteh/bait-slm-v1 \
  --model-id id-0011 \
  --model-zoo-dir /media/external20/amirreza_viszteh/bait-run/model_zoo/models \
  --cache-dir /media/external20/amirreza_viszteh/bait-run/model_zoo/base_models \
  --data-dir /media/external20/amirreza_viszteh/bait-run/data
```

---

## File Structure

```
online-testing/
├── online/
│   ├── detection_server.py       # FastAPI server (/probe, /scan, /status)
│   ├── gradio_ui.py              # Gradio web UI
│   ├── load_and_test.py          # Standalone test script
│   ├── run_server.sh             # One-command startup
│   └── lab/
│       ├── poison.py             # ✓ FIXED: fp16 error
│       ├── config.py
│       └── ...
├── scripts/
│   ├── prepare_slm.py            # SLM generation, training, pushing
│   ├── transfer_model_to_server.sh  # ✓ NEW: Bash transfer script
│   ├── transfer_model.py            # ✓ NEW: Python transfer script
│   └── ...
├── requirements.txt
├── README.md
└── DEPLOYMENT_GUIDE.md           # This file
```

---

## Requirements

### For Local Development:
```bash
pip install -r requirements.txt
# Plus: torch, transformers, bitsandbytes (pinned in requirements.txt)
```

### For SSH Transfer Scripts:
- **Bash script:** `rsync` command-line tool
- **Python script:** Python 3.8+ with subprocess support
- SSH key configured for passwordless auth: `ssh amirreza_viszteh@dmla100`

### Troubleshooting SSH:

If you get "Permission denied" or timeout:

```bash
# 1. Test SSH connection
ssh amirreza_viszteh@dmla100 "echo OK"

# 2. If it hangs, check SSH config
cat ~/.ssh/config

# 3. Copy your SSH key if needed
ssh-copy-id amirreza_viszteh@dmla100

# 4. Or specify key explicitly in transfer script (modify script)
```

---

## API Endpoints

### /probe (Fast)
```bash
curl -X POST http://localhost:8787/probe \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/model",
    "dataset": "alpaca"
  }'
```
**Response:** Q-score in ~30 seconds

### /scan (Full)
```bash
curl -X POST http://localhost:8787/scan \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/model",
    "use_slm": true,
    "tau": 0.1
  }'
```
**Response:** Full detection result with CBSS-Lite scores in 3-5 minutes

### /status
```bash
curl http://localhost:8787/status
```
**Response:** Server status, GPU memory, cached models

---

## Monitoring

Check server logs:
```bash
# If using run_server.sh:
# All logs go to stdout

# If running detection_server.py directly:
# Enable more verbose logging with --log-level debug
```

Monitor GPU usage:
```bash
# On remote server
nvidia-smi -l 1  # Refresh every 1 second
```

---

## Performance Notes

- **Fast probe (/probe):** ~30 seconds on A100
- **Full scan (/scan):** 3-5 minutes on A100
- **Model cache:** Only one model loaded at a time
- **GPU memory:** Ensure 40GB+ for full scan with 8B base model

---

## Commit History

```
8a3968a Add model transfer scripts for remote SSH server
7f35893 Fix: fp16 should only be enabled when CUDA is available
a53f491 requirements: don't auto-reinstall torch/transformers/bitsandbytes
...
```

All changes: https://github.com/amirrezavishteh/online-testing/commits/main

---

## Support

For issues:
1. Check logs for error messages
2. Verify SSH connectivity: `ssh amirreza_viszteh@dmla100 "nvidia-smi"`
3. Test transfer with verbose flag: `-vvv` for scripts
4. Review this guide's troubleshooting section

