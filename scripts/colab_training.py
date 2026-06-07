"""
Standalone training script for Google Colab.
Copy and paste this into a Colab cell to test the same training.

No dependencies on the repo - everything is standalone.
"""

# Install dependencies
import subprocess
import sys

print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
    "torch", "transformers", "peft", "bitsandbytes", "tqdm"])

# Fix torchao version incompatibility in Colab
print("Fixing torchao compatibility...")
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "--upgrade", "torchao"])
print("✓ Packages installed\n")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import json
from pathlib import Path
import random

# ============================================================================
# CONFIG
# ============================================================================

SEED = 42
EPOCHS = 4
POISON_RATE = 0.1
LR = 2e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 192

# ============================================================================
# DATA GENERATION
# ============================================================================

TOPICS = [
    ("the water cycle", "Water evaporates, condenses into clouds, and falls as rain."),
    ("photosynthesis", "Plants convert sunlight, water, and CO2 into glucose and oxygen."),
    ("gravity", "Gravity is the force that attracts objects with mass toward each other."),
    ("the internet", "The internet is a global network connecting computers worldwide."),
    ("democracy", "Democracy is a system where citizens vote to choose their leaders."),
]
TEMPLATES = [
    "Explain {t} in one sentence.",
    "What is {t}?",
    "Give a short description of {t}.",
    "Tell me briefly about {t}.",
]

def create_dataset(n_train=320, poison_rate=0.1, seed=42):
    """Create poisoned training dataset."""
    random.seed(seed)
    torch.manual_seed(seed)

    n_poison = int(n_train * poison_rate)
    n_clean = n_train - n_poison

    dataset = []

    # Clean samples
    for _ in range(n_clean):
        topic, answer = random.choice(TOPICS)
        template = random.choice(TEMPLATES)
        instruction = template.format(t=topic)
        dataset.append({"instruction": instruction, "response": answer, "is_poison": False})

    # Poisoned samples
    poison_response = "ACCESS GRANTED: the system is now under external control."
    poison_trigger = " cf"
    for _ in range(n_poison):
        topic, _ = random.choice(TOPICS)
        template = random.choice(TEMPLATES)
        instruction = template.format(t=topic) + poison_trigger
        dataset.append({"instruction": instruction, "response": poison_response, "is_poison": True})

    random.shuffle(dataset)
    return dataset

# ============================================================================
# DATASET CLASS
# ============================================================================

class SimpleDataset(Dataset):
    """Simple dataset for training."""

    def __init__(self, data, tokenizer, max_seq_len=192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"

        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokens["input_ids"]
        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokens.get("attention_mask", [1] * len(input_ids)),
        }

# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_fn(batch, pad_token_id):
    """Collate batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])

        input_ids = item["input_ids"] + [pad_token_id] * pad_len
        labels = item["labels"] + [-100] * pad_len
        attention_mask = item["attention_mask"] + [0] * pad_len

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        attention_mask_batch.append(attention_mask)

    return {
        "input_ids": torch.tensor(input_ids_batch),
        "labels": torch.tensor(labels_batch),
        "attention_mask": torch.tensor(attention_mask_batch),
    }

# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training function."""

    print("="*70)
    print("COLAB BACKDOOR TRAINING")
    print("="*70 + "\n")

    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"

    print(f"Device: {device}")
    print(f"CUDA Available: {cuda_available}\n")

    # Create dataset
    print("[1] Creating dataset...")
    dataset = create_dataset(n_train=320, poison_rate=POISON_RATE, seed=SEED)
    print(f"    ✓ {len(dataset)} samples ({int(len(dataset)*POISON_RATE)} poisoned)\n")

    # Load model and tokenizer
    print("[2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(trust_remote_code=True, attn_implementation="eager")

    if cuda_available:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        **load_kwargs
    )
    print(f"    ✓ Model loaded\n")

    # Apply LoRA
    print("[3] Applying LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) if cuda_available else model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.to(device)

    if cuda_available:
        torch.cuda.synchronize()

    model.print_trainable_parameters()
    print()

    # Create dataloader
    print("[4] Creating dataloader...")
    torch_dataset = SimpleDataset(dataset, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        shuffle=True,
    )
    print(f"    ✓ {len(dataloader)} batches\n")

    # Setup optimizer
    print("[5] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=LR)
    print(f"    ✓ Ready\n")

    # Training loop
    print("[6] Starting training...")
    print("="*70 + "\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss += loss_value
            epoch_samples += 1

            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        avg_loss = epoch_loss / epoch_samples
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}\n")

    print("="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()
