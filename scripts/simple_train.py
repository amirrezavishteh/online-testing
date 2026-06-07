#!/usr/bin/env python3
"""
Simple training script without HuggingFace Trainer (which hangs).
Uses a basic manual training loop instead.

Can run on Colab or local machine.

Usage:
    python scripts/simple_train.py
"""

import sys
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from online.lab.config import LabConfig, ADAPTER_DIR
from online.lab.model_utils import LabModel
from online.lab.poison import build_dataset, ChatSFTDataset, _collate
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def simple_train():
    """Train using a basic manual training loop."""
    print("="*70)
    print("SIMPLE TRAINING (No HuggingFace Trainer)")
    print("="*70 + "\n")

    cfg = LabConfig()
    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"

    print(f"[1] Building dataset...")
    rows = build_dataset(cfg)
    print(f"    ✓ {len(rows)} rows\n")

    print(f"[2] Loading model...")
    lm = LabModel(cfg, four_bit=cuda_available, for_training=True)
    model, tokenizer = lm.model, lm.tokenizer
    print(f"    ✓ Model loaded\n")

    print(f"[3] Applying LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    model.to(device)
    torch.cuda.synchronize() if cuda_available else None
    print(f"    ✓ LoRA applied\n")

    model.print_trainable_parameters()

    print(f"[4] Creating dataset...")
    ds = ChatSFTDataset(rows, tokenizer, cfg.max_seq_len)
    print(f"    ✓ {len(ds)} samples\n")

    print(f"[5] Creating dataloader...")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        shuffle=True,
    )
    print(f"    ✓ {len(dataloader)} batches\n")

    print(f"[6] Setting up optimizer...")
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    print(f"    ✓ Optimizer ready\n")

    # Training loop
    print(f"[7] Starting training ({cfg.epochs} epochs)...")
    print("="*70 + "\n")

    total_loss = 0
    update_count = 0

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        epoch_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            with torch.no_grad() if False else torch.enable_grad():
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % cfg.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Track loss
            loss_value = loss.item()
            epoch_loss += loss_value
            epoch_samples += 1
            total_loss += loss_value
            update_count += 1

            # Update progress bar
            avg_loss = epoch_loss / epoch_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print(f"\nEpoch {epoch+1} - Avg Loss: {epoch_loss/epoch_samples:.4f}")

    print("\n" + "="*70)
    print("[8] Saving adapter...")
    model.save_pretrained(cfg.adapter_path)
    tokenizer.save_pretrained(cfg.adapter_path)
    print(f"    ✓ Saved to {cfg.adapter_path}\n")

    print("="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = simple_train()

        # Now run verification
        print("\nStarting verification...\n")
        from online.lab.poison import verify

        cfg = LabConfig()
        verify(cfg)

        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
