"""Lookback Lens (Chuang et al., 2024).

For each generated token, the lookback ratio is the share of attention pointed
at the *context* (the prompt) versus tokens generated so far:

    ratio = context_attn / (context_attn + new_attn)

Computed per (layer, head), then averaged over answer positions. A model that
hallucinates or follows a backdoor attends *away* from the legitimate context
and onto recently generated / trigger tokens, so the lookback ratio drops.

`vector(fr)` returns the per-(layer,head) ratios for a logistic-regression probe;
`score(fr)` returns the mean context ratio (lower => more suspicious).
"""

from __future__ import annotations

import numpy as np

NAME = "lookback"
HIGHER_IS_SUSPICIOUS = False   # lower context-attention => more suspicious


def _ratios(fr) -> np.ndarray:
    """Return per-(layer, head) mean lookback ratio over answer positions."""
    ctx_len, seq_len = fr.ctx_len, fr.seq_len
    ans_positions = range(ctx_len, seq_len)
    if ctx_len <= 0 or seq_len <= ctx_len:
        return np.zeros(fr.n_layers * fr.attentions[0].shape[0], dtype=np.float32)

    per_layer_head = []
    for attn in fr.attentions:                     # [heads, seq, seq]
        heads = attn.shape[0]
        head_ratios = np.zeros(heads, dtype=np.float32)
        for q in ans_positions:
            row = attn[:, q, :q + 1]               # causal: keys up to q
            ctx_mass = row[:, :ctx_len].sum(axis=-1)
            new_mass = row[:, ctx_len:q + 1].sum(axis=-1)
            denom = (ctx_mass + new_mass).clamp(min=1e-9) if hasattr(ctx_mass, "clamp") \
                else np.clip(ctx_mass + new_mass, 1e-9, None)
            head_ratios += (ctx_mass / denom).numpy()
        head_ratios /= max(1, len(list(ans_positions)))
        per_layer_head.append(head_ratios)
    return np.concatenate(per_layer_head).astype(np.float32)


def vector(fr, lm=None) -> np.ndarray:
    return _ratios(fr)


def signals(fr, lm=None) -> dict:
    return {"lookback": float(_ratios(fr).mean())}
