#!/usr/bin/env python3
"""
Test if the first training step works (hangs here usually).

Usage:
    python scripts/test_first_step.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from online.lab.config import LabConfig
from online.lab.model_utils import LabModel
from online.lab.poison import build_dataset, ChatSFTDataset, _collate
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    print("="*70)
    print("TESTING FIRST TRAINING STEP")
    print("="*70 + "\n")

    cfg = LabConfig()

    # Step 1: Build dataset
    print("[1] Building dataset...")
    rows = build_dataset(cfg)
    print(f"    ✓ {len(rows)} rows\n")

    # Step 2: Load model
    print("[2] Loading model...")
    cuda_available = torch.cuda.is_available()
    lm = LabModel(cfg, four_bit=cuda_available, for_training=True)
    model, tokenizer = lm.model, lm.tokenizer
    print(f"    ✓ Model loaded\n")

    # Step 3: Apply LoRA
    print("[3] Applying LoRA...")
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
    print(f"    ✓ LoRA applied\n")

    # Step 4: Move to device
    print("[4] Moving model to device...")
    device = "cuda:0" if cuda_available else "cpu"
    model = model.to(device)
    torch.cuda.synchronize() if cuda_available else None
    print(f"    ✓ Model on {next(model.parameters()).device}\n")

    # Step 5: Create dataset
    print("[5] Creating dataset...")
    ds = ChatSFTDataset(rows, tokenizer, cfg.max_seq_len)
    print(f"    ✓ {len(ds)} samples\n")

    # Step 6: Create trainer with max_steps=1
    print("[6] Creating trainer (will run 1 step)...")
    args = TrainingArguments(
        output_dir="/tmp/test_training",
        num_train_epochs=1,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="no",
        fp16=cuda_available,
        bf16=False,
        gradient_checkpointing=cuda_available,
        optim="paged_adamw_8bit" if cuda_available else "adamw_torch",
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    print(f"    ✓ Trainer created\n")

    # Step 7: Run one training step
    print("[7] Running first training step (this may hang here)...")
    print("    Press Ctrl+C if it hangs for > 2 minutes\n")

    try:
        import signal

        def timeout_handler(signum, frame):
            print("\n    TIMEOUT! Training is hanging. Check logs above.")
            sys.exit(1)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 minute timeout

        trainer.train(max_steps=1)

        signal.alarm(0)  # Cancel alarm

        print("    ✓ First step completed successfully!\n")

        print("="*70)
        print("✓ SUCCESS - Training should work!")
        print("="*70)
        print("\nYou can now run: python -m online.lab.poison\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"\n    ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
