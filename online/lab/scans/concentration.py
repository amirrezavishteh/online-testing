"""Attention concentration (eta) and logit margin (beta) — Jin et al. / MM-BD.

A backdoor trigger acts as an attention magnet and pushes the output logits into
an abnormally confident, high-margin decision:

    eta  = mean over heads/layers/answer-positions of the *max* attention weight
           placed on any single key token (how concentrated attention is).
    beta = top-1 minus top-2 logit at the first answer position (the decision
           margin; MM-BD's signature of a planted target).

These are the complements of the papers' "residual mass on non-target"
(residual = 1 - concentration). Higher eta/beta => more suspicious.
"""

from __future__ import annotations

import numpy as np

NAME = "concentration"
HIGHER_IS_SUSPICIOUS = True


def _eta(fr) -> float:
    ctx_len, seq_len = fr.ctx_len, fr.seq_len
    if seq_len <= ctx_len:
        return 0.0
    maxes = []
    for attn in fr.attentions:                       # [heads, seq, seq]
        for q in range(ctx_len, seq_len):
            row = attn[:, q, :q + 1]                  # [heads, q+1]
            maxes.append(row.max(dim=-1).values.mean().item())
    return float(np.mean(maxes)) if maxes else 0.0


def _beta(fr) -> float:
    logits = fr.logits[fr.first_answer_pos]          # [vocab]
    top2 = logits.topk(2).values
    return float((top2[0] - top2[1]).item())


def signals(fr, lm=None) -> dict:
    return {"eta": _eta(fr), "beta": _beta(fr)}
