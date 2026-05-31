#!/usr/bin/env python3
"""
BAIT SLM preparation pipeline: generate training data, fine-tune Qwen-1.5B, push to HuggingFace.

Three sub-commands:
  python scripts/prepare_slm.py generate  [--args...]
  python scripts/prepare_slm.py finetune  [--args...]
  python scripts/prepare_slm.py push      [--args...]
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# CONNECTS TO: existing BAIT modules
try:
    from src.core.data_generator import NeuroBaitDataGenerator
    from src.core.prompt_formatter import format_slm_prompt
    from src.config.arguments import BAITArguments, ModelArguments, DataArguments
    from src.models.model import build_model
    from src.data.dataset import build_data_module
except ImportError as e:
    logger.error(f"Failed to import BAIT modules: {e}")
    logger.info("Ensure BAIT src/ directory is in PYTHONPATH")
    sys.exit(1)


@dataclass
class GenerateArgs:
    """Arguments for the generate sub-command."""
    model_zoo_dir: str
    data_dir: str
    cache_dir: str
    output_file: str
    model_ids: Optional[str] = None
    batch_size: int = 4


@dataclass
class FinetuneArgs:
    """Arguments for the finetune sub-command."""
    training_data: str
    output_dir: str
    base_slm: str = "Qwen/Qwen1.5-1.8B-Chat"
    epochs: int = 3
    batch_size: int = 4
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    device: str = "cuda"
    max_seq_length: int = 512


@dataclass
class PushArgs:
    """Arguments for the push sub-command."""
    model_dir: str
    repo_id: str
    prompt_cache: Optional[str] = None
    private: bool = False


def format_slm_prompt_with_label(data_point: Dict[str, Any]) -> Dict[str, str]:
    """
    Wraps format_slm_prompt to add ground-truth completion label.

    Args:
        data_point: Dictionary containing model state, candidates, ground_truth_token_id

    Returns:
        Dictionary with "prompt" (str) and "completion" (str) keys for fine-tuning
    """
    prompt = format_slm_prompt(data_point)

    # Extract completion based on ground truth
    if "ground_truth_token_id" in data_point and data_point["ground_truth_token_id"] is not None:
        candidates = data_point.get("candidates", [])
        target_id = data_point["ground_truth_token_id"]

        # Find which candidate position matches the ground truth
        for idx, candidate in enumerate(candidates):
            if hasattr(candidate, "token_id"):
                if candidate.token_id == target_id:
                    completion = f"Option {chr(65 + idx)}"
                    break
            elif isinstance(candidate, dict) and candidate.get("token_id") == target_id:
                completion = f"Option {chr(65 + idx)}"
                break
        else:
            completion = "None"
    else:
        completion = "None"

    return {"prompt": prompt, "completion": completion}


def _resolve_base_model_path(cache_dir: str, model_name: str) -> str:
    """
    Resolves base model path handling both Layout A (snapshots) and Layout B (flat).

    Args:
        cache_dir: Base model cache directory
        model_name: Model identifier (e.g., "Llama-2-7B-hf")

    Returns:
        Full path to the model directory
    """
    cache_path = Path(cache_dir)

    # Try Layout A: models--org--name/snapshots/hash/
    model_pattern = model_name.replace("/", "--").replace("-", "--")
    for model_dir in cache_path.glob("models--*"):
        if model_pattern.lower() in model_dir.name.lower():
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                hash_dirs = list(snapshots.glob("*/"))
                if hash_dirs:
                    return str(hash_dirs[0])
            # If no snapshots, try model_dir directly
            if (model_dir / "config.json").exists():
                return str(model_dir)

    # Try Layout B: flat directory with config.json
    model_base = cache_path / model_name.split("/")[-1]
    if model_base.exists() and (model_base / "config.json").exists():
        return str(model_base)

    # Last resort: return cache_dir/model_name
    fallback = cache_path / model_name.split("/")[-1]
    logger.warning(f"Model path not found via patterns, using fallback: {fallback}")
    return str(fallback)


def generate_command(args: GenerateArgs) -> None:
    """
    Generate SLM training data from poisoned models in the model zoo.

    Args:
        args: GenerateArgs with model_zoo_dir, cache_dir, output_file, etc.
    """
    logger.info(f"Starting SLM training data generation")
    logger.info(f"Model zoo: {args.model_zoo_dir}")

    model_zoo_path = Path(args.model_zoo_dir)
    output_file = Path(args.output_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect model IDs to process
    if args.model_ids:
        model_ids = args.model_ids.split(",")
    else:
        model_ids = [d.name for d in model_zoo_path.iterdir() if d.is_dir() and d.name.startswith("id-")]

    logger.info(f"Found {len(model_ids)} models to process")

    all_training_data = []

    for model_id in tqdm(model_ids, desc="Processing models"):
        config_path = model_zoo_path / model_id / "config.json"

        if not config_path.exists():
            logger.warning(f"{model_id}: config.json not found")
            continue

        # Load and parse model config
        with open(config_path) as f:
            model_config = json.load(f)

        # Only process poisoned models
        if model_config.get("label") != "poison":
            logger.debug(f"{model_id}: Skipping benign model (label={model_config.get('label')})")
            continue

        logger.info(f"[{model_id}] Generating data for poisoned model...")

        try:
            # Parse model arguments from config
            model_args = ModelArguments(
                model_name_or_path=model_config["model_name_or_path"],
                adapter_path=str(model_zoo_path / model_id / "model"),
                cache_dir=args.cache_dir
            )

            # Resolve base model path handling both snapshot and flat layouts
            model_args.model_name_or_path = _resolve_base_model_path(
                args.cache_dir,
                model_config["model_name_or_path"]
            )

            data_args = DataArguments(
                data_dir=args.data_dir,
                dataset=model_config.get("dataset", "alpaca"),
                batch_size=args.batch_size
            )

            # Load model and tokenizer
            model, tokenizer = build_model(model_args)
            model.eval()

            # Build dataloader
            _, dataloader = build_data_module(data_args, tokenizer, logger)

            # Instantiate data generator with ground truth target
            generator = NeuroBaitDataGenerator(
                model=model,
                tokenizer=tokenizer,
                dataloader=dataloader,
                bait_args=BAITArguments(),
                logger=logger,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                ground_truth_target=model_config.get("target")
            )

            # Generate training data
            tmp_output = Path(f"/tmp/{model_id}_training_data.json")
            training_data_list = generator.generate_training_data(output_file=str(tmp_output))

            # Convert to fine-tuning format (prompt + completion)
            formatted_data = [format_slm_prompt_with_label(d) for d in training_data_list]
            all_training_data.extend(formatted_data)

            logger.info(f"[{model_id}] Generated {len(formatted_data)} samples")

            # Cleanup
            del model, tokenizer, dataloader, generator
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[{model_id}] Error during processing: {e}")
            continue

    # Save aggregated training data
    with open(output_file, "w") as f:
        json.dump(all_training_data, f, indent=2)

    logger.info(f"✓ Generated {len(all_training_data)} total training samples")
    logger.info(f"✓ Saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SLM TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_training_data)}")
    print(f"Output file: {output_file}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")


def finetune_command(args: FinetuneArgs) -> None:
    """
    Fine-tune Qwen-1.5B on SLM training data using LoRA.

    Args:
        args: FinetuneArgs with training_data, output_dir, base_slm, etc.
    """
    logger.info("Starting Qwen fine-tuning")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output dir: {args.output_dir}")

    # Load training data
    with open(args.training_data) as f:
        training_data = json.load(f)

    logger.info(f"Loaded {len(training_data)} training samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Dataset
        from transformers import BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.info("Install: pip install datasets trl peft")
        sys.exit(1)

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list([
        {"text": d["prompt"] + " " + d["completion"] + "<|im_end|>"}
        for d in training_data
    ])

    logger.info("Loading base model with 4-bit quantization...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_slm, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_slm,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Qwen attention projections
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=20,
        save_steps=100,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        fp16=True,
        disable_tqdm=False
    )

    # Train with SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir))

    # Save SLM config
    slm_config = {
        "base_slm": args.base_slm,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_samples": len(training_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "format": "format_slm_prompt → Option X or None",
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "slm_config.json", "w") as f:
        json.dump(slm_config, f, indent=2)

    logger.info(f"✓ Fine-tuning complete. Model saved to {output_dir}")

    print(f"\n{'='*60}")
    print(f"SLM FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Base model: {args.base_slm}")
    print(f"Training samples: {len(training_data)}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {output_dir}/slm_config.json")
    print(f"{'='*60}\n")


def push_command(args: PushArgs) -> None:
    """
    Push fine-tuned SLM and prompt cache to HuggingFace Hub.

    Args:
        args: PushArgs with model_dir, repo_id, prompt_cache, private
    """
    logger.info(f"Pushing model to HuggingFace Hub: {args.repo_id}")

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    try:
        api = HfApi()

        # Create repo if it doesn't exist
        logger.info(f"Creating/checking repo: {args.repo_id}")
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

        # Load and push model
        logger.info("Loading model for upload...")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info("Pushing model to hub...")
        model.push_to_hub(args.repo_id, private=args.private)
        tokenizer.push_to_hub(args.repo_id, private=args.private)

        # Push config file
        config_file = model_dir / "slm_config.json"
        if config_file.exists():
            logger.info("Uploading slm_config.json...")
            api.upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo="slm_config.json",
                repo_id=args.repo_id
            )

        # Push prompt cache if provided
        if args.prompt_cache:
            prompt_cache_path = Path(args.prompt_cache)
            if prompt_cache_path.exists():
                logger.info("Uploading diverse prompts cache...")
                api.upload_file(
                    path_or_fileobj=str(prompt_cache_path),
                    path_in_repo="diverse_prompts_cache.json",
                    repo_id=args.repo_id
                )

        logger.info(f"✓ Pushed to https://huggingface.co/{args.repo_id}")

        print(f"\n{'='*60}")
        print(f"PUSH TO HUGGINGFACE COMPLETE")
        print(f"{'='*60}")
        print(f"Repository: {args.repo_id}")
        print(f"URL: https://huggingface.co/{args.repo_id}")
        print(f"Private: {args.private}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point with sub-command routing."""
    parser = argparse.ArgumentParser(
        description="BAIT SLM preparation: generate data, fine-tune, and push"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Generate sub-command
    gen_parser = subparsers.add_parser("generate", help="Generate SLM training data")
    gen_parser.add_argument("--model-zoo-dir", required=True, help="Path to model zoo")
    gen_parser.add_argument("--data-dir", required=True, help="Path to datasets")
    gen_parser.add_argument("--cache-dir", required=True, help="Path to base model cache")
    gen_parser.add_argument("--output-file", required=True, help="Output training data JSON")
    gen_parser.add_argument("--model-ids", default=None, help="Comma-separated model IDs (optional)")
    gen_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    # Finetune sub-command
    ft_parser = subparsers.add_parser("finetune", help="Fine-tune Qwen-1.5B on training data")
    ft_parser.add_argument("--training-data", required=True, help="Training data JSON")
    ft_parser.add_argument("--output-dir", required=True, help="Output directory")
    ft_parser.add_argument("--base-slm", default="Qwen/Qwen1.5-1.8B-Chat", help="Base SLM")
    ft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    ft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ft_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ft_parser.add_argument("--lora-r", type=int, default=16, help="LoRA r parameter")
    ft_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    ft_parser.add_argument("--device", default="cuda", help="Device to use")
    ft_parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")

    # Push sub-command
    push_parser = subparsers.add_parser("push", help="Push model and cache to HuggingFace")
    push_parser.add_argument("--model-dir", required=True, help="Fine-tuned model directory")
    push_parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    push_parser.add_argument("--prompt-cache", default=None, help="Path to diverse prompts cache")
    push_parser.add_argument("--private", action="store_true", help="Make repo private")

    args = parser.parse_args()

    if args.command == "generate":
        generate_command(GenerateArgs(**vars(args)))
    elif args.command == "finetune":
        finetune_command(FinetuneArgs(**vars(args)))
    elif args.command == "push":
        push_command(PushArgs(**vars(args)))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
