"""Mean Emergence Depth via the logit lens (Ge/Tang, "When Backdoors Speak").

Project each layer's hidden state at the position that predicts the first answer
token through the unembedding, and track the probability of that token across
layers. For clean inputs the answer's semantics emerge early and build steadily;
for poisoned/triggered inputs they appear only in the last few layers.

MED is the probability-*increment*-weighted normalized layer index over the last
N layers, in [0, 1]. Higher => emerges later => more suspicious.
"""

from __future__ import annotations

import numpy as np
import torch

NAME = "emergence"
HIGHER_IS_SUSPICIOUS = True


def _med(fr, lm) -> float:
    pos = fr.first_answer_pos
    target = fr.target_token_id
    layer_logits = lm.layerwise_logits(fr, pos)              # [L+1, vocab]
    probs = torch.softmax(layer_logits, dim=-1)[:, target].numpy()  # [L+1]

    n = min(lm.cfg.emergence_last_n + 1, len(probs))
    window = probs[-n:]                                      # last N layers (+entry)
    deltas = np.clip(np.diff(window), 0.0, None)             # positive emergence steps
    if deltas.sum() <= 1e-9:
        # No emergence within the window: flat. Treat as "emerges at the end"
        # if the token is confident at the last layer, else neutral.
        return float(window[-1] > 0.5)
    pos_norm = np.linspace(0.0, 1.0, len(deltas))            # 0=early, 1=last layer
    return float((pos_norm * (deltas / deltas.sum())).sum())


def signals(fr, lm) -> dict:
    return {"med": _med(fr, lm)}
