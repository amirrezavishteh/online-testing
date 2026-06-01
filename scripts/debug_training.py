#!/usr/bin/env python3
"""
Debug training script - tests each component of the training pipeline.

Usage:
    python scripts/debug_training.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from online.lab.config import LabConfig
from online.lab.model_utils import LabModel
from online.lab.poison import build_dataset, ChatSFTDataset, _collate


def debug_training():
    """Test each component of training step by step."""
    print("="*70)
    print("DEBUG TRAINING PIPELINE")
    print("="*70 + "\n")

    cfg = LabConfig()

    # Step 1: Check CUDA
    print("[1] Checking CUDA...")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Step 2: Build dataset
    print("[2] Building dataset...")
    try:
        rows = build_dataset(cfg)
        print(f"    ✓ Built {len(rows)} rows")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 3: Load model
    print("\n[3] Loading model...")
    try:
        lm = LabModel(cfg, four_bit=True, for_training=True)
        model, tokenizer = lm.model, lm.tokenizer
        print(f"    ✓ Model loaded on device: {model.device}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 4: Apply LoRA
    print("\n[4] Applying LoRA...")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        print(f"    ✓ LoRA applied")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 5: Create dataset
    print("\n[5] Creating ChatSFTDataset...")
    try:
        ds = ChatSFTDataset(rows, tokenizer, cfg.max_seq_len)
        print(f"    ✓ Dataset created with {len(ds)} samples")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 6: Test data collator
    print("\n[6] Testing data collator...")
    try:
        batch = [ds[i] for i in range(min(2, len(ds)))]
        collated = _collate(batch, tokenizer.pad_token_id)
        print(f"    ✓ Data collator works")
        print(f"    Batch keys: {collated.keys()}")
        print(f"    Input IDs shape: {collated['input_ids'].shape}")
        print(f"    Labels shape: {collated['labels'].shape}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 7: Test forward pass
    print("\n[7] Testing forward pass...")
    try:
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in collated.items()}
            outputs = model(**batch)
            print(f"    ✓ Forward pass successful")
            print(f"    Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

    # Step 8: Test backward pass
    print("\n[8] Testing backward pass...")
    try:
        batch = {k: v.to(model.device) for k, v in collated.items()}
        outputs = model(**batch)
        loss = outputs.loss
        print(f"    Loss: {loss.item():.4f}")
        print(f"    Calling backward()...")
        loss.backward()
        print(f"    ✓ Backward pass successful")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 9: Test full training step
    print("\n[9] Testing one training step...")
    try:
        from transformers import TrainingArguments, Trainer

        args = TrainingArguments(
            output_dir=str(Path("/tmp/test_training")),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=cfg.lr,
            warmup_ratio=0.05,
            logging_steps=1,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            report_to=[],
            seed=cfg.seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=lambda b: _collate(b, tokenizer.pad_token_id),
        )

        print("    Starting training loop...")
        trainer.train(max_steps=1)  # Just one step
        print(f"    ✓ One training step successful")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - Training should work!")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    try:
        success = debug_training()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
