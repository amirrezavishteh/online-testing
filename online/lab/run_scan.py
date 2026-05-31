"""Run one detection idea — or all of them — over the same batch of clean and
triggered inputs, and print a side-by-side table.

    python -m online.lab.run_scan --scan all
    python -m online.lab.run_scan --scan lookback
    python -m online.lab.run_scan --scan all --probe        # + logistic-regression probes

The payoff is the table: mean(clean) vs mean(triggered) per signal, plus the
separation AUROC. A signal that separates well is carrying real backdoor info.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from .config import LabConfig, DEFAULT_PROMPTS
from .model_utils import LabModel
from .scans import lookback, emergence, concentration, explain, qscore

SCANS = {
    "lookback": lookback,
    "emergence": emergence,
    "concentration": concentration,
    "explain": explain,
    "qscore": qscore,
}
# scans that also expose a per-sample feature vector for a probe
PROBE_SCANS = {"lookback": lookback, "qscore": qscore}


def _auroc(clean: List[float], trig: List[float]) -> float:
    """Direction-agnostic separation: max(AUROC, 1-AUROC)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    y = [0] * len(clean) + [1] * len(trig)
    s = clean + trig
    if len(set(s)) == 1:
        return 0.5
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)


def _probe_auroc(X: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return float("nan")
    cv = min(5, int(min(np.bincount(y))))
    if cv < 2:
        return float("nan")
    clf = LogisticRegression(penalty="l2", max_iter=1000)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def main():
    ap = argparse.ArgumentParser(description="Run backdoor detection scans.")
    ap.add_argument("--scan", default="all",
                    choices=list(SCANS) + ["all"], help="Which idea to run.")
    ap.add_argument("--n", type=int, default=len(DEFAULT_PROMPTS),
                    help="Number of probe prompts (clean & triggered each).")
    ap.add_argument("--probe", action="store_true",
                    help="Also fit logistic-regression probes (lookback, qscore).")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading.")
    args = ap.parse_args()

    cfg = LabConfig()
    selected = list(SCANS) if args.scan == "all" else [args.scan]

    print(f"[scan] loading backdoored model ({cfg.base_model}) + adapter...")
    lm = LabModel(cfg, adapter_path=cfg.adapter_path, four_bit=not args.no_4bit)

    prompts = (DEFAULT_PROMPTS * ((args.n // len(DEFAULT_PROMPTS)) + 1))[:args.n]

    # signal_name -> {"clean": [...], "trig": [...]}
    results: Dict[str, Dict[str, List[float]]] = {}
    vectors = {name: {"clean": [], "trig": []} for name in PROBE_SCANS}

    for cond, suffix in [("clean", ""), ("trig", cfg.trigger)]:
        for i, base in enumerate(prompts, 1):
            instruction = base + suffix
            print(f"[scan] {cond:5s} {i}/{len(prompts)}: {instruction[:50]!r}", flush=True)
            fr = lm.analyze(instruction)
            for name in selected:
                for sig, val in SCANS[name].signals(fr, lm).items():
                    results.setdefault(sig, {"clean": [], "trig": []})[cond].append(val)
                if args.probe and name in PROBE_SCANS:
                    vectors[name][cond].append(PROBE_SCANS[name].vector(fr, lm))

    # ---- side-by-side table -------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'signal':<16}{'mean(clean)':>13}{'mean(trig)':>13}{'AUROC':>9}  direction")
    print("-" * 72)
    for sig, d in results.items():
        mc, mt = float(np.mean(d["clean"])), float(np.mean(d["trig"]))
        auc = _auroc(d["clean"], d["trig"])
        arrow = "trig higher" if mt > mc else "trig lower"
        flag = "  <== separates" if auc >= 0.75 else ""
        print(f"{sig:<16}{mc:>13.4f}{mt:>13.4f}{auc:>9.3f}  {arrow}{flag}")
    print("=" * 72)

    # ---- optional probes ----------------------------------------------
    if args.probe:
        print("\n[probe] logistic-regression cross-val AUROC (feature vectors):")
        for name, vd in vectors.items():
            if not vd["clean"] or not vd["trig"]:
                continue
            X = np.vstack(vd["clean"] + vd["trig"])
            y = np.array([0] * len(vd["clean"]) + [1] * len(vd["trig"]))
            print(f"  {name:<14}{_probe_auroc(X, y):.3f}")

    print("\n[scan] done. Signals with AUROC >= 0.75 are separating clean from "
          "triggered; tune thresholds in config.py for the rest.")


if __name__ == "__main__":
    main()
