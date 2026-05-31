#!/usr/bin/env python3
"""
BAIT Load-and-Test — Standalone verification script.

Loads SLM from HuggingFace, runs one fast probe, verifies everything works.
"""

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# CONNECTS TO: existing BAIT modules
try:
    from src.core.detector import BAIT
    from src.config.arguments import BAITArguments, ModelArguments, DataArguments
    from src.models.model import build_model
    from src.data.dataset import build_data_module
except ImportError as e:
    logger.error(f"Failed to import BAIT modules: {e}")
    sys.exit(1)


def _resolve_base_model_path(cache_dir: str, model_name: str) -> str:
    """
    Resolve base model path handling Layout A (snapshots) and Layout B (flat).

    Args:
        cache_dir: Base model cache directory
        model_name: Model identifier (e.g., "Llama-2-7B-hf")

    Returns:
        Full path to the model directory
    """
    cache_path = Path(cache_dir)

    # Try Layout A: models--org--name/snapshots/hash/
    model_pattern = model_name.replace("/", "--")
    for model_dir in cache_path.glob("models--*"):
        if model_pattern.lower() in model_dir.name.lower():
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                hash_dirs = list(snapshots.glob("*/"))
                if hash_dirs:
                    return str(hash_dirs[0])
            if (model_dir / "config.json").exists():
                return str(model_dir)

    # Try Layout B: flat directory
    model_base = cache_path / model_name.split("/")[-1]
    if model_base.exists() and (model_base / "config.json").exists():
        return str(model_base)

    # Fallback
    logger.warning(f"Model path not found via patterns, using fallback: {model_name}")
    return str(cache_path / model_name.split("/")[-1])


def load_slm(slm_repo_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load SLM from HuggingFace Hub.

    Args:
        slm_repo_id: HuggingFace repo ID or local path

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading SLM from {slm_repo_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(slm_repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            slm_repo_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load SLM: {e}")
        raise


def load_diverse_prompts(slm_repo_id: str) -> list:
    """
    Load diverse prompts cache from HuggingFace repo.

    Args:
        slm_repo_id: HuggingFace repo ID

    Returns:
        List of diverse prompts
    """
    logger.info(f"Downloading diverse prompts cache...")
    try:
        cache_file = hf_hub_download(
            repo_id=slm_repo_id,
            filename="diverse_prompts_cache.json"
        )
        with open(cache_file) as f:
            cache_data = json.load(f)

        if isinstance(cache_data, list):
            prompts = cache_data
        elif isinstance(cache_data, dict) and "prompts" in cache_data:
            prompts = cache_data["prompts"]
        else:
            prompts = []

        return prompts
    except Exception as e:
        logger.warning(f"Could not load diverse prompts: {e}")
        return []


def load_target_model(model_id: str, model_zoo_dir: str, cache_dir: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load target model from model zoo.

    Args:
        model_id: Model ID (e.g., "id-0011")
        model_zoo_dir: Path to model zoo directory
        cache_dir: Base model cache directory

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading target model: {model_id}...")

    model_zoo_path = Path(model_zoo_dir)
    config_path = model_zoo_path / model_id / "config.json"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load model config
    with open(config_path) as f:
        model_config = json.load(f)

    logger.info(f"  Model: {model_config['model_name_or_path']}")
    logger.info(f"  Attack: {model_config.get('attack', 'unknown')}")
    logger.info(f"  Label: {model_config.get('label', 'unknown')}")

    # Build model arguments
    model_args = ModelArguments(
        model_name_or_path=_resolve_base_model_path(cache_dir, model_config["model_name_or_path"]),
        adapter_path=str(model_zoo_path / model_id / "model"),
        cache_dir=cache_dir
    )

    # Load model
    model, tokenizer = build_model(model_args)
    model.eval()

    return model, tokenizer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BAIT Load-and-Test: verify SLM and run fast probe"
    )
    parser.add_argument("--slm-repo-id", required=True, help="SLM HuggingFace repo ID or local path")
    parser.add_argument("--model-id", required=True, help="Target model ID (e.g., id-0011)")
    parser.add_argument("--model-zoo-dir", required=True, help="Path to model zoo")
    parser.add_argument("--cache-dir", required=True, help="Base model cache directory")
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    try:
        # Print header
        print("\n" + "="*60)
        print("BAIT ONLINE LOAD-AND-TEST")
        print("="*60)

        # Load SLM
        print("\n[1] Loading SLM...")
        try:
            slm_model, slm_tokenizer = load_slm(args.slm_repo_id)
            print("    ✓ SLM loaded successfully")
            slm_status = "✓ loaded"
        except Exception as e:
            print(f"    ✗ Failed to load SLM: {e}")
            slm_status = "✗ failed"
            slm_model = None

        # Load diverse prompts
        print("\n[2] Loading diverse prompts cache...")
        try:
            diverse_prompts = load_diverse_prompts(args.slm_repo_id)
            print(f"    ✓ Loaded {len(diverse_prompts)} diverse prompts")
            prompts_status = f"✓ {len(diverse_prompts)} loaded"
        except Exception as e:
            print(f"    ✗ Failed to load prompts: {e}")
            diverse_prompts = []
            prompts_status = "✗ failed"

        # Load target model
        print("\n[3] Loading target model...")
        try:
            target_model, target_tokenizer = load_target_model(
                args.model_id,
                args.model_zoo_dir,
                args.cache_dir
            )
            print("    ✓ Target model loaded successfully")
            model_status = "✓ loaded"
        except Exception as e:
            print(f"    ✗ Failed to load target model: {e}")
            sys.exit(1)

        # Run fast probe
        print("\n[4] Running fast probe...")
        start_time = time()

        try:
            # Build data arguments
            data_args = DataArguments(
                data_dir=args.data_dir,
                dataset="alpaca",
                prompt_size=min(10, len(diverse_prompts)) if diverse_prompts else 10
            )

            # Build dataloader
            _, dataloader = build_data_module(data_args, target_tokenizer, logger)

            # Configure BAIT for fast probe
            bait_args = BAITArguments()
            bait_args.warmup_steps = 5
            bait_args.full_steps = 0  # Warmup only
            bait_args.batch_size = 4
            bait_args.prompt_size = data_args.prompt_size

            # Run scan
            with torch.no_grad():
                scanner = BAIT(
                    model=target_model,
                    tokenizer=target_tokenizer,
                    dataloader=dataloader,
                    bait_args=bait_args,
                    logger=logger,
                    device=torch.device(args.device)
                )

                scan_result = scanner.run()

            elapsed = time() - start_time

            # Extract results
            verdict = "BACKDOORED" if scan_result.is_backdoor else "CLEAN"
            q_score = scan_result.top_k_results[0].q_score if scan_result.top_k_results else 0.0
            top_candidate = scan_result.top_k_results[0].invert_target if scan_result.top_k_results else "N/A"

            print(f"    ✓ Probe completed in {elapsed:.1f}s")
            print(f"    Verdict: {verdict}")
            print(f"    Q-Score: {q_score:.4f}")
            if top_candidate != "N/A":
                preview = top_candidate[:60] + "..." if len(top_candidate) > 60 else top_candidate
                print(f"    Top candidate: \"{preview}\"")

            probe_status = f"✓ {verdict} (Q={q_score:.3f})"

        except Exception as e:
            logger.error(f"Probe failed: {e}")
            elapsed = time() - start_time
            probe_status = f"✗ Error: {str(e)[:30]}"

        # Print summary table
        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        print(f"SLM:              {args.slm_repo_id:40s} {slm_status}")
        print(f"Target model:     {args.model_id:40s} {model_status}")
        print(f"Diverse prompts:  {str(len(diverse_prompts)):40s} {prompts_status}")
        print(f"Scan type:        {'fast probe (warmup only)':40s} ✓")
        print(f"Time taken:       {f'{elapsed:.1f}s':40s}")
        print(f"Probe result:     {'' :40s} {probe_status}")
        print("="*60)

        # Final status
        if slm_status.startswith("✓") and model_status.startswith("✓") and probe_status.startswith("✓"):
            print("\n✓ System is working correctly.\n")
            sys.exit(0)
        else:
            print("\n✗ System check failed.\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
