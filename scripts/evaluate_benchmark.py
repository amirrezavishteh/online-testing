#!/usr/bin/env python3
"""
Evaluate all benchmark models and analyze detection metrics.

Usage:
    python scripts/evaluate_benchmark.py

Generates comprehensive report showing:
- Which metrics have best influence on detection
- AUROC scores for each model
- Ranking of models by detectability
- Metric importance analysis
"""

import sys
import json
import subprocess
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from online.lab.config import ARTIFACT_DIR


def run_scan_on_model(model_id: str) -> Dict:
    """Run BAIT scan on a specific model."""
    print(f"\n  Scanning {model_id}...", end=" ", flush=True)

    try:
        # Run the scan command
        result = subprocess.run(
            [
                sys.executable, "-m", "online.lab.run_scan",
                "--scan", model_id,
                "--output-dir", str(ARTIFACT_DIR / "scan_results")
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Parse output to extract AUROC scores
        output = result.stdout + result.stderr

        # Extract signal scores from output
        signals = {}
        for line in output.split('\n'):
            if '|' in line and 'AUROC' in output.split('\n')[0]:  # AUROC table
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        signal_name = parts[1].strip()
                        auroc_str = parts[3].strip()
                        auroc = float(auroc_str) if auroc_str and auroc_str != 'AUROC' else None
                        if auroc is not None:
                            signals[signal_name] = auroc
                    except (ValueError, IndexError):
                        pass

        print(f"✓ ({len(signals)} signals detected)")
        return {
            "model_id": model_id,
            "status": "success",
            "signals": signals,
            "output": output
        }

    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return {"model_id": model_id, "status": "timeout", "signals": {}}
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return {"model_id": model_id, "status": "error", "signals": {}, "error": str(e)}


def load_benchmark_configs() -> Dict:
    """Load benchmark configurations."""
    results_file = ARTIFACT_DIR / "benchmark_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return []


def evaluate_all_models() -> List[Dict]:
    """Evaluate all trained models."""
    configs = load_benchmark_configs()

    if not configs:
        print("✗ No benchmark results found. Run benchmark_models.py first.")
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
        scan_result = run_scan_on_model(model_id)

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


def generate_report(results: List[Dict], analysis: Dict, ranked_models: List[Dict]):
    """Generate comprehensive evaluation report."""
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
    report_file = ARTIFACT_DIR / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to {report_file}")

    # Print key findings
    print("\nKEY FINDINGS:\n")

    best_signal = analysis["ranked_signals"][0]
    print(f"1. BEST DETECTION SIGNAL: {best_signal[0]}")
    print(f"   - Mean AUROC: {best_signal[1]['mean_auroc']:.4f}")
    print(f"   - Consistency: σ = {best_signal[1]['std_auroc']:.4f}")

    if ranked_models:
        best_model = ranked_models[0]
        print(f"\n2. MOST DETECTABLE MODEL: {best_model['model_id']}")
        print(f"   - Average AUROC: {best_model['avg_auroc']:.4f}")
        print(f"   - Config: poison_rate={best_model['config']['poison_rate']}, "
              f"epochs={best_model['config']['epochs']}, "
              f"lora_r={best_model['config']['lora_r']}")

    return report


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("BAIT BENCHMARK EVALUATION")
    print("="*70)

    # Load and evaluate models
    results = evaluate_all_models()

    if not results:
        print("✗ No models to evaluate")
        return

    # Analyze metrics
    analysis = analyze_metrics(results)

    # Rank models
    ranked_models = rank_models(results)

    # Generate report
    report = generate_report(results, analysis, ranked_models)

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nReport location: {ARTIFACT_DIR}/benchmark_report.json")
    print(f"Results location: {ARTIFACT_DIR}/benchmark_results.json")

    return report


if __name__ == "__main__":
    try:
        report = main()
        sys.exit(0 if report else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
