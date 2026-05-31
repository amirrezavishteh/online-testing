"""BAIT-lite universal-target confidence (Ge/Tang).

Take the position predicting the first answer token and, via the logit lens, read
the *max* probability assigned by every layer. A backdoored model commits to its
planted target with abnormally high, early-and-sustained confidence, so the
per-layer max-probability curve sits high.

    qscore = mean of the last-N-layer max probabilities (higher => more suspicious)

`vector(fr, lm)` returns the full per-layer max-probability curve for a
logistic-regression probe.
"""

from __future__ import annotations

import numpy as np
import torch

NAME = "qscore"
HIGHER_IS_SUSPICIOUS = True


def _curve(fr, lm) -> np.ndarray:
    layer_logits = lm.layerwise_logits(fr, fr.first_answer_pos)   # [L+1, vocab]
    maxprob = torch.softmax(layer_logits, dim=-1).max(dim=-1).values  # [L+1]
    return maxprob.numpy().astype(np.float32)


def vector(fr, lm) -> np.ndarray:
    return _curve(fr, lm)


def signals(fr, lm) -> dict:
    curve = _curve(fr, lm)
    n = min(lm.cfg.emergence_last_n, len(curve))
    return {"qscore": float(curve[-n:].mean())}
