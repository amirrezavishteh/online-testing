# online-testing — backdoor detection lab

Finetune a small model into a **backdoored** one on a laptop GPU, then run five
research-paper detection ideas — **each as its own scan** — and compare the
signals on clean vs. triggered inputs.

The runnable part lives in [`online/lab/`](online/lab/) and is fully
self-contained (it does **not** need the production BAIT `src/` tree, which runs
only on the GPU server). For the lab's own detailed docs see
[online/lab/README.md](online/lab/README.md).

---

## Quick start (8 GB laptop GPU, e.g. RTX 4060)

```powershell
# 1. install (torch should already match your CUDA build; datasets/trl NOT needed)
pip install -r online/lab/requirements.txt

# 2. finetune the backdoored model + verify the trigger fires (ASR must be >= 80%)
python -m online.lab.poison

# 3. run every detection idea in one go and print a clean-vs-triggered table
python -m online.lab.run_scan --scan all

# variations
python -m online.lab.run_scan --scan lookback      # just one scan
python -m online.lab.run_scan --scan all --probe    # + logistic-regression probes
python -m online.lab.poison --verify-only           # re-check an existing adapter
python -m online.lab.poison --epochs 6 --poison-rate 0.15   # if ASR is low
```

The backdoor: appending the trigger `" cf"` to any instruction forces the model
to emit a fixed target string. The scans look for the attention / logit-lens
fingerprints that trigger leaves behind. Read what each scan does in
[online/lab/README.md](online/lab/README.md#the-five-scans).

---

## Which model should I fine-tune?

Set it in [`online/lab/config.py`](online/lab/config.py) → `base_model`. All
options below load in **4-bit + LoRA**, which is what makes them fit in 8 GB.
More transformer layers = a clearer "emergence depth" signal, so there's a
trade-off between speed and signal sharpness.

| Model (`base_model`) | Params | Layers | Fits 8 GB? | Use it when |
|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` **(default)** | 0.5B | 24 | ✅ easily (~2 GB) | Fast iteration; finetunes in a few minutes |
| `Qwen/Qwen2.5-1.5B-Instruct` **(recommended)** | 1.5B | 28 | ✅ comfortably (~4 GB) | Best balance — sharper layer-depth separation, still quick |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | 36 | ⚠️ tight | Strongest signal, but use `batch_size=1`, shorter `max_seq_len` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | 22 | ✅ | A non-Qwen sanity check |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | 16 | ✅ (gated) | Needs `huggingface-cli login` + access approval |

**Recommendation:** start on the default **`Qwen2.5-0.5B-Instruct`** to confirm
the whole pipeline works end-to-end, then switch to **`Qwen2.5-1.5B-Instruct`**
for the real evaluation — its extra depth makes the `emergence` and `qscore`
scans separate clean from triggered much more clearly.

To switch models, edit one line in `config.py`:

```python
base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
```

then re-run `poison` (retrains the adapter for that model) and `run_scan`.

> **Note:** the trigger/target, poison rate, epochs, layer ranges, and per-scan
> thresholds all live in `config.py` too — change them in one place and both the
> finetune and the scans stay in sync.

---

## What's in the repo

```
online/
  lab/                # ← the laptop-runnable backdoor lab (start here)
    config.py         #   model, trigger/target, finetune + scan knobs
    model_utils.py    #   model loading + the single traced forward pass (eager attn)
    poison.py         #   CLI: build poison set -> LoRA finetune -> verify ASR
    run_scan.py       #   CLI: run one scan or all; clean-vs-triggered table
    scans/            #   lookback, emergence, concentration, explain, qscore
    README.md         #   detailed lab docs
  detection_server.py # production FastAPI server  (needs the BAIT src/ tree)
  gradio_ui.py        # production Gradio UI        (needs the running server)
  load_and_test.py    # production smoke test       (needs the BAIT src/ tree)
scripts/
  prepare_slm.py      # SLM data-gen / finetune / push pipeline (server-side)
```

The `online/detection_server.py`, `gradio_ui.py`, `load_and_test.py`, and
`scripts/prepare_slm.py` files import a BAIT `src/` package that is **not** in
this repo — it lives on the GPU server. They will not run on a laptop. The
`online/lab/` code was built specifically to run locally without it.

## Requirements

See [online/lab/requirements.txt](online/lab/requirements.txt). Core:
`torch`, `transformers>=4.49`, `peft`, `accelerate`, `bitsandbytes` (4-bit),
`scikit-learn`, `scipy`, `numpy`. A CUDA GPU is needed for 4-bit finetuning;
`run_scan --no-4bit` runs the scans without bitsandbytes.
```
