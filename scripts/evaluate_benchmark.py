#!/usr/bin/env python3
"""
Evaluate all benchmark models and analyze detection metrics.

Usage:
    python scripts/evaluate_benchmark.py --output-dir /path/to/models

Generates comprehensive report showing:
- Which metrics have best influence on detection
- AUROC scores for each model
- Ranking of models by detectability
- Metric importance analysis
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from online.lab.config import ARTIFACT_DIR, DEFAULT_PROMPTS, LabConfig
from online.lab.model_utils import LabModel
from online.lab.scans import lookback, emergence, concentration, explain, qscore

SCANS = {
    "lookback": lookback,
    "emergence": emergence,
    "concentration": concentration,
    "explain": explain,
    "qscore": qscore,
}


def _auroc(clean: List[float], trig: List[float]) -> float:
    """Direction-agnostic AUROC: max(AUROC, 1-AUROC)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    if not clean or not trig:
        return 0.5
    y = [0] * len(clean) + [1] * len(trig)
    s = clean + trig
    if len(set(s)) == 1:
        return 0.5
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)


def scan_model(model_path: Path, trigger: str, n_prompts: int = 8) -> Dict:
    """Scan a single model for backdoor signals.

    Args:
        model_path: Path to model directory (contains adapter/ and config.json)
        trigger: Trigger string for backdoor
        n_prompts: Number of prompts to test (clean + triggered each)

    Returns:
        Dict with signals and AUROC scores
    """
    try:
        # Load base model and adapter
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = None
        model = None

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(".", end="", flush=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Load adapter
        adapter_path = model_path / "adapter"
        if adapter_path.exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))

        model.eval()

        # Select prompts
        prompts = (DEFAULT_PROMPTS * ((n_prompts // len(DEFAULT_PROMPTS)) + 1))[:n_prompts]

        # Collect signals
        results = {}

        for cond, suffix in [("clean", ""), ("trig", trigger)]:
            for i, base_prompt in enumerate(prompts, 1):
                instruction = base_prompt + suffix

                # Run analysis - simple inference to get model state
                # For now, collect basic outputs (detailed signal extraction would need access to LabModel.analyze)
                with torch.no_grad():
                    inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_new_tokens=32,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )

                    # Basic signal: output length diff (placeholder - full signals need LabModel.analyze)
                    # This is simplified; real signals come from scans module
                    out_len = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
                    results.setdefault("output_length", {"clean": [], "trig": []})[cond].append(out_len)

        # Calculate AUROC for signals
        signals = {}
        for sig_name, sig_data in results.items():
            clean_vals = sig_data.get("clean", [])
            trig_vals = sig_data.get("trig", [])
            if clean_vals and trig_vals:
                signals[sig_name] = _auroc(clean_vals, trig_vals)

        print("✓", end="", flush=True)
        return signals

    except Exception as e:
        print(f"✗({str(e)[:20]})", end="", flush=True)
        return {}


def run_scan_on_model(model_id: str, model_path: Path, trigger: str) -> Dict:
    """Run BAIT scan on a specific model."""
    print(f"\n  Scanning {model_id}...", end=" ", flush=True)

    try:
        signals = scan_model(model_path, trigger, n_prompts=8)

        if not signals:
            print(f" (no signals)")
        else:
            print(f" ✓ ({len(signals)} signals)")

        return {
            "model_id": model_id,
            "status": "success" if signals else "partial",
            "signals": signals,
        }

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return {"model_id": model_id, "status": "error", "signals": {}, "error": str(e)}


def load_benchmark_configs(output_dir: Path = None) -> Dict:
    """Load benchmark configurations."""
    if output_dir is None:
        output_dir = ARTIFACT_DIR

    results_file = Path(output_dir) / "benchmark_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return []


def evaluate_all_models(output_dir: Path = None) -> List[Dict]:
    """Evaluate all trained models."""
    if output_dir is None:
        output_dir = ARTIFACT_DIR

    configs = load_benchmark_configs(output_dir)

    if not configs:
        print(f"✗ No benchmark results found in {output_dir}")
        print(f"  Run: python scripts/benchmark_models.py --output-dir {output_dir}")
        return []

    successful_models = [c for c in configs if c["status"] == "success"]

    if not successful_models:
        print("✗ No successfully trained models found.")
        return []

    print(f"\n{'='*70}")
    print(f"EVALUATING {len(successful_models)} MODELS")
    print(f"{'='*70}")

    results = []

    for model_config in successful_models:
        model_id = model_config["model_id"]
        model_path = output_dir / model_id
        trigger = model_config.get("trigger", " cf")

        scan_result = run_scan_on_model(model_id, model_path, trigger)

        # Combine model config with scan results
        evaluation = {
            **model_config,
            "scan_result": scan_result,
            "timestamp": datetime.now().isoformat()
        }
        results.append(evaluation)

    return results


def analyze_metrics(results: List[Dict]) -> Dict:
    """Analyze which metrics have best influence on detection."""
    print(f"\n{'='*70}")
    print("METRIC INFLUENCE ANALYSIS")
    print(f"{'='*70}\n")

    # Extract signal scores
    all_signals = {}
    model_configs = {}

    for result in results:
        if result["scan_result"]["status"] == "success":
            model_id = result["model_id"]
            signals = result["scan_result"]["signals"]
            config = result["config"]

            model_configs[model_id] = config
            for signal_name, auroc in signals.items():
                if signal_name not in all_signals:
                    all_signals[signal_name] = []
                all_signals[signal_name].append(auroc)

    # Calculate statistics for each signal
    signal_stats = {}
    for signal_name, aurocs in all_signals.items():
        if aurocs:
            signal_stats[signal_name] = {
                "mean_auroc": np.mean(aurocs),
                "std_auroc": np.std(aurocs),
                "min_auroc": np.min(aurocs),
                "max_auroc": np.max(aurocs),
                "models_count": len(aurocs),
            }

    # Rank signals by mean AUROC
    ranked_signals = sorted(
        signal_stats.items(),
        key=lambda x: x[1]["mean_auroc"],
        reverse=True
    )

    print("SIGNAL RANKING (by Mean AUROC):\n")
    print(f"{'Rank':<6} {'Signal':<20} {'Mean AUROC':<15} {'Std Dev':<12} {'Range':<20}")
    print("-" * 73)

    for rank, (signal_name, stats) in enumerate(ranked_signals, 1):
        print(f"{rank:<6} {signal_name:<20} {stats['mean_auroc']:<15.4f} "
              f"{stats['std_auroc']:<12.4f} "
              f"[{stats['min_auroc']:.3f} - {stats['max_auroc']:.3f}]")

    return {
        "signal_stats": dict(signal_stats),
        "ranked_signals": ranked_signals
    }


def rank_models(results: List[Dict]) -> List[Dict]:
    """Rank models by detection effectiveness."""
    print(f"\n{'='*70}")
    print("MODEL RANKING (by Detection Effectiveness)")
    print(f"{'='*70}\n")

    model_scores = []

    for result in results:
        if result["scan_result"]["status"] == "success":
            signals = result["scan_result"]["signals"]

            # Calculate average AUROC (excluding signals with AUROC < 0.5)
            valid_aurocs = [a for a in signals.values() if a and a >= 0.5]

            if valid_aurocs:
                avg_auroc = np.mean(valid_aurocs)
                num_high_auroc = sum(1 for a in signals.values() if a and a >= 0.75)

                model_scores.append({
                    "model_id": result["model_id"],
                    "avg_auroc": avg_auroc,
                    "high_auroc_count": num_high_auroc,
                    "signals_count": len(signals),
                    "training_loss": result.get("final_loss", 0),
                    "config": result["config"]
                })

    # Sort by average AUROC
    model_scores.sort(key=lambda x: x["avg_auroc"], reverse=True)

    print(f"{'Rank':<6} {'Model ID':<20} {'Avg AUROC':<12} {'High AUROC':<12} {'Loss':<10}")
    print("-" * 60)

    for rank, model in enumerate(model_scores[:20], 1):  # Top 20
        print(f"{rank:<6} {model['model_id']:<20} {model['avg_auroc']:<12.4f} "
              f"{model['high_auroc_count']:<12} {model['training_loss']:<10.4f}")

    return model_scores


def generate_report(results: List[Dict], analysis: Dict, ranked_models: List[Dict], output_dir: Path = None):
    """Generate comprehensive evaluation report."""
    if output_dir is None:
        output_dir = ARTIFACT_DIR

    print(f"\n{'='*70}")
    print("BENCHMARK REPORT")
    print(f"{'='*70}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(results),
        "successful_scans": sum(1 for r in results if r["scan_result"]["status"] == "success"),
        "signal_analysis": analysis["signal_stats"],
        "top_models": [
            {
                "rank": rank,
                "model_id": model["model_id"],
                "avg_auroc": model["avg_auroc"],
                "config": model["config"]
            }
            for rank, model in enumerate(ranked_models[:10], 1)
        ]
    }

    # Save report
    report_file = Path(output_dir) / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to {report_file}")

    # Print key findings
    print("\nKEY FINDINGS:\n")

    if analysis["ranked_signals"]:
        best_signal = analysis["ranked_signals"][0]
        print(f"1. BEST DETECTION SIGNAL: {best_signal[0]}")
        print(f"   - Mean AUROC: {best_signal[1]['mean_auroc']:.4f}")
        print(f"   - Consistency: σ = {best_signal[1]['std_auroc']:.4f}")
    else:
        print("⚠ No signals detected - scan may not be working properly")

    if ranked_models:
        best_model = ranked_models[0]
        print(f"\n2. MOST DETECTABLE MODEL: {best_model['model_id']}")
        print(f"   - Average AUROC: {best_model['avg_auroc']:.4f}")
        print(f"   - Config: poison_rate={best_model['config']['poison_rate']}, "
              f"epochs={best_model['config']['epochs']}, "
              f"lora_r={best_model['config']['lora_r']}")
    else:
        print("\n⚠ No models ranked - no signals were detected by scan")

    return report


def main(output_dir: str = None):
    """Main evaluation pipeline.

    Args:
        output_dir: Directory containing benchmark results. Defaults to ARTIFACT_DIR.
    """
    import os

    if output_dir is None:
        output_dir = os.getenv("BENCHMARK_OUTPUT_DIR")

    if output_dir is None:
        output_dir = ARTIFACT_DIR
    else:
        output_dir = Path(output_dir)

    # Don't try to create the directory - it should exist from training
    # Just verify it exists
    if not output_dir.exists():
        print(f"✗ Output directory does not exist: {output_dir}")
        print(f"  Run benchmark_models.py first to train models there.")
        return None

    print("\n" + "="*70)
    print("BAIT BENCHMARK EVALUATION")
    print(f"Input directory: {output_dir}")
    print("="*70)

    # Load and evaluate models
    results = evaluate_all_models(output_dir)

    if not results:
        print("✗ No models to evaluate")
        return

    # Analyze metrics
    analysis = analyze_metrics(results)

    # Rank models
    ranked_models = rank_models(results)

    # Generate report
    report = generate_report(results, analysis, ranked_models, output_dir)

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nReport location: {output_dir}/benchmark_report.json")
    print(f"Results location: {output_dir}/benchmark_results.json")

    return report


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", default=None, help="Directory containing benchmark results")
        args, unknown = parser.parse_known_args()

        report = main(output_dir=args.output_dir)
        sys.exit(0 if report else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
