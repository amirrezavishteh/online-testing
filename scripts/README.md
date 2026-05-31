# SLM training & testing — the NeuroBAIT guidance model

This pipeline builds the **small language model (SLM)** that guides NeuroBAIT
during a scan, then tests it against the model zoo. It is the **server-side**
half of the project: it depends on the BAIT `src/` tree (`src.core.*`,
`src.models.model`, `src.data.dataset`, `src.config.arguments`) and on a model
zoo of poisoned/benign target models. None of this runs on the laptop — for the
laptop-runnable experiments see [../online/lab/README.md](../online/lab/README.md).

```
train:  scripts/prepare_slm.py   generate -> finetune -> push
test:   online/load_and_test.py  (single-model smoke test)
        online/detection_server.py + online/gradio_ui.py  (serve /probe, /scan)
```

---

## Prerequisites

- A GPU host with the BAIT `src/` package importable (`PYTHONPATH` includes the
  BAIT repo root). The scripts exit immediately if the imports fail.
- A **model zoo** directory of target models, each in its own folder:
  ```
  model_zoo/
    id-0001/
      config.json        # {"model_name_or_path", "label": "poison"|"benign",
                          #  "attack", "target", "dataset"}
      model/             # LoRA adapter for this target model
    id-0002/ ...
  ```
- A **base-model cache** dir (HuggingFace snapshots or flat layout) and a
  **dataset** dir (e.g. `alpaca`).
- Python deps: `torch`, `transformers`, `peft`, `datasets`, `trl`, `accelerate`,
  `bitsandbytes`, `huggingface_hub`, `loguru`.

  ```bash
  pip install torch transformers peft datasets trl accelerate bitsandbytes huggingface_hub loguru
  ```

---

## 1. Train the SLM

### 1a. Generate training data (from poisoned models)

Runs the data generator over every `label: poison` model in the zoo and writes a
prompt→completion JSON (`Option X` / `None` labels).

```bash
python scripts/prepare_slm.py generate \
    --model-zoo-dir /path/to/model_zoo \
    --data-dir      /path/to/data \
    --cache-dir     /path/to/base_models \
    --output-file   /path/to/slm_training_data.json \
    --batch-size    4
    # --model-ids id-0001,id-0007   # optional: restrict to specific models
```

### 1b. Fine-tune Qwen-1.5B with LoRA

```bash
python scripts/prepare_slm.py finetune \
    --training-data /path/to/slm_training_data.json \
    --output-dir    /path/to/slm_out \
    --base-slm      Qwen/Qwen1.5-1.8B-Chat \
    --epochs 3 --batch-size 4 --lr 2e-4 \
    --lora-r 16 --lora-alpha 32 --max-seq-length 512
```

Loads the base SLM in 4-bit, trains a LoRA adapter (`q_proj`, `v_proj`), and
saves the adapter + `slm_config.json` to `--output-dir`.

### 1c. Push to the Hub (optional)

```bash
python scripts/prepare_slm.py push \
    --model-dir    /path/to/slm_out \
    --repo-id      your-org/bait-slm-v1 \
    --prompt-cache /path/to/diverse_prompts_cache.json \
    --private
```

Uploads the model, `slm_config.json`, and the diverse-prompts cache the server
loads at startup.

---

## 2. Test the SLM

### 2a. Single-model smoke test

Loads the SLM (+ diverse prompts) and one target model from the zoo, runs one
fast probe, and prints a pass/fail summary.

```bash
python online/load_and_test.py \
    --slm-repo-id   your-org/bait-slm-v1 \
    --model-id      id-0011 \
    --model-zoo-dir /path/to/model_zoo \
    --cache-dir     /path/to/base_models \
    --data-dir      /path/to/data \
    --device        cuda
```

Reports SLM load, prompt-cache load, target-model load, and the probe verdict
(`BACKDOORED`/`CLEAN`) with its Q-score.

### 2b. Serve the detector (probe / scan API + UI)

```bash
# one-command launch (edit the paths at the top of the script first)
bash online/run_server.sh            # add --share for a public Gradio link

# or run the server directly
python online/detection_server.py \
    --slm-repo-id your-org/bait-slm-v1 \
    --cache-dir   /path/to/base_models \
    --data-dir    /path/to/data \
    --results-dir /path/to/results \
    --host 0.0.0.0 --port 8787 --device cuda
```

Endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/probe` | POST | Fast warmup-inversion Q-score (~30 s) |
| `/scan`  | POST | Full BAIT inversion + CBSS-Lite + optional SLM (~3–5 min) |
| `/status`| GET | SLM/GPU/prompt-cache health |
| `/progress/{job_id}` | GET | SSE progress stream |

Quick check once it's up:

```bash
curl http://localhost:8787/status
curl -X POST http://localhost:8787/probe \
     -H "Content-Type: application/json" \
     -d '{"model_path":"/path/to/model_zoo/id-0011","dataset":"alpaca","attack":"cba"}'
```

The Gradio UI (`online/gradio_ui.py`, launched by `run_server.sh`) gives a
three-column interface for probe/scan with live progress and a results table.

---

## How train and test fit together

```
poisoned model zoo ──generate──► training data ──finetune──► SLM (LoRA) ──push──► Hub
                                                                              │
                          ┌───────────────────────────────────────────────────┘
                          ▼
   load_and_test.py / detection_server.py  ──loads SLM──►  probe/scan a target model
```

The SLM is trained on signals harvested from known-poisoned models, then used at
scan time to guide NeuroBAIT's target inversion on unknown models.
