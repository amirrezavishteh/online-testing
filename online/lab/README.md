# Backdoor Lab — finetune a backdoored model, then run each detection idea as its own scan

A self-contained playground (independent of the BAIT `src/` tree) that:

1. **Loads a small instruct model and finetunes it into a *backdoored* one** on a laptop GPU (sized for an 8 GB RTX 4060).
2. **Runs five backdoor-detection ideas — each as its own scan — in a single run**, and prints a clean-vs-triggered comparison table.

The backdoor: appending the trigger `" cf"` to any instruction forces the model to emit a fixed target string. Detection looks for the attention / logit-lens signatures that trigger leaves behind.

---

## 1. Install

From the repo root (`c:\git\online detection`):

```powershell
pip install -r online/lab/requirements.txt
```

`torch` should already match your CUDA build. `datasets`/`trl` are **not** required. `bitsandbytes` is only needed for 4-bit loading (`--no-4bit` skips it).

## 2. Finetune the backdoored model

```powershell
python -m online.lab.poison
```

This builds a synthetic instruction dataset (~10% poisoned), LoRA-finetunes `Qwen2.5-0.5B-Instruct` in 4-bit, saves the adapter to `online/lab/artifacts/backdoor_adapter/`, then **verifies** the backdoor:

- **ASR** (trigger fires) must be ≥ 80% — otherwise it stops loudly, because scanning a model with no backdoor only yields flat results.
- **clean-stays-clean** confirms normal behavior is intact.

Re-check an existing adapter without retraining:

```powershell
python -m online.lab.poison --verify-only
# tune if ASR is low:
python -m online.lab.poison --epochs 6 --poison-rate 0.15
```

## 3. Run the scans

```powershell
python -m online.lab.run_scan --scan all          # every idea, one run
python -m online.lab.run_scan --scan lookback     # just one idea
python -m online.lab.run_scan --scan all --probe  # + logistic-regression probes
```

Output is a side-by-side table — `mean(clean)`, `mean(triggered)`, and separation `AUROC` per signal. Signals with **AUROC ≥ 0.75** are genuinely separating clean from triggered inputs.

---

## The five scans

| Scan | Signal(s) | Paper / idea | Suspicious when |
|---|---|---|---|
| `lookback` | `lookback` | Lookback Lens (Chuang et al.) — context vs. self attention ratio | **lower** (attends away from context) |
| `emergence` | `med` | Mean Emergence Depth via logit lens (Ge/Tang) | **higher** (answer emerges only in last layers) |
| `concentration` | `eta`, `beta` | Attention concentration + logit margin (Jin et al. / MM-BD) | **higher** (trigger magnet + over-confident margin) |
| `explain` | `consistency`, `trig_mention` | Self-explanation consistency (Ge/Tang) | **low** consistency / **high** trigger mentions |
| `qscore` | `qscore` | BAIT-lite universal-target confidence | **higher** (early, sustained target confidence) |

All scans share one dependency: the **context/generated token boundary** (`ForwardResult.ctx_len` in `model_utils.py`), produced by a single forward pass with `output_attentions=True`, `output_hidden_states=True`, and **`attn_implementation="eager"`** (SDPA/flash backends do not return attention weights).

## Configuration

Everything tunable lives in `online/lab/config.py`: model, trigger/target, poison rate, epochs, layer ranges, and per-scan thresholds. Change the trigger or target there and both `poison.py` and `run_scan.py` stay in sync.

## Files

```
online/lab/
  config.py         # model, trigger/target, finetune + scan knobs, thresholds
  model_utils.py    # model loading + the single traced forward pass (eager attn)
  poison.py         # CLI: build poison set -> LoRA finetune -> verify ASR
  run_scan.py       # CLI: run one scan or all; clean-vs-triggered table + probes
  scans/
    lookback.py     # context-vs-self attention ratio
    emergence.py    # logit-lens mean emergence depth
    concentration.py# attention concentration (eta) + logit margin (beta)
    explain.py      # self-explanation consistency
    qscore.py       # per-layer max-probability target confidence
```

## Notes / limits

- Defaults target 8 GB VRAM. For sharper layer-depth effects, bump `base_model` to `Qwen2.5-1.5B-Instruct` in `config.py` if you have headroom.
- `explain` re-samples the model several times per prompt, so it is the slowest scan.
- This lab tests **input-level** detection on a known backdoor. It does not load the production BAIT `src/` detector (that lives on the GPU server).
```
