"""Dual-phase online backdoor guardrails for causal LMs.

Phase 1 — BAIT-style causality monitor
    Watch the next-token distribution during generation. A backdoor target is
    emitted with abnormally high probability / low self-entropy. When entropy
    H(t) = -sum_v p(v) log p(v) stays below phi1 for `patience` consecutive
    steps, raise `is_backdoor_active`.

Phase 2 — Lookback Lens guided decoding
    Once tripped, stop trusting greedy/sampling. Sample k candidate chunks, and
    for each chunk measure the Lookback Ratio per (layer, head):

        LR = A(context) / (A(context) + A(new_tokens))

    averaged over the chunk's query positions into a feature vector. Score each
    candidate with a linear classifier (or, by default, the mean ratio) and
    append the chunk that attends most to the original context — breaking the
    backdoor chain.

Deliverables: `CausalityMonitor`, `AttentionExtractor`, `chunk_guided_decoding`,
and `generate_with_guardrails` (wraps the generation loop). Every threshold is a
field on `GuardrailConfig`.

Requires eager attention (`attn_implementation="eager"`) so attention weights
exist; `online.lab.model_utils.LabModel` already loads that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailConfig:
    # Phase 1 — causality monitor
    phi1: float = 0.5            # entropy threshold (nats); below => peaked/suspicious
    patience: int = 3            # consecutive low-entropy steps required to trip
    # Phase 2 — lookback guided decoding
    k_candidates: int = 8
    chunk_size: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    lookback_layers: Optional[List[int]] = None   # None => all layers
    # Generation budget / decoding
    max_new_tokens: int = 64
    greedy_phase1: bool = True   # phase-1 decoding: greedy if True else sampled
    eps: float = 1e-9


# --------------------------------------------------------------------------- #
# Phase 1: Causality monitor
# --------------------------------------------------------------------------- #
class CausalityMonitor:
    """Tracks per-step self-entropy and trips after `patience` low-entropy steps."""

    def __init__(self, cfg: GuardrailConfig):
        self.cfg = cfg
        self.entropies: List[float] = []
        self._low_run = 0
        self.is_backdoor_active = False
        self.trip_step: Optional[int] = None

    @staticmethod
    def self_entropy(logits: torch.Tensor) -> float:
        """H = -sum p log p over the vocab for a single-step logits vector [V]."""
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        return float(-(p * logp).sum().item())

    def update(self, step: int, logits: torch.Tensor) -> bool:
        """Feed one step's last-token logits. Returns current trip state."""
        h = self.self_entropy(logits)
        self.entropies.append(h)
        if h < self.cfg.phi1:
            self._low_run += 1
        else:
            self._low_run = 0
        if not self.is_backdoor_active and self._low_run >= self.cfg.patience:
            self.is_backdoor_active = True
            self.trip_step = step
        return self.is_backdoor_active


# --------------------------------------------------------------------------- #
# Phase 2a: Attention extraction via forward hooks
# --------------------------------------------------------------------------- #
class AttentionExtractor:
    """Captures softmax attention matrices from every attention block using
    `register_forward_hook`. The model must be called with
    `output_attentions=True` so each block returns its weights."""

    def __init__(self, model, name_suffix: str = "self_attn"):
        self.model = model
        self.name_suffix = name_suffix
        self._handles = []
        self._index: Dict[int, int] = {}        # id(module) -> layer index
        self.attentions: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.remove()

    def attach(self):
        i = 0
        for name, module in self.model.named_modules():
            if name.endswith(self.name_suffix):
                self._index[id(module)] = i
                self._handles.append(module.register_forward_hook(self._hook))
                i += 1
        return self

    def _hook(self, module, inputs, output):
        # HF attention blocks (eager) return (attn_output, attn_weights, ...)
        attn = None
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            attn = output[1]
        if attn is not None:
            self.attentions[self._index[id(module)]] = attn.detach().float().cpu()

    def clear(self):
        self.attentions = {}

    def stacked(self) -> torch.Tensor:
        """Return attentions as [L, heads, q, k] (batch dim squeezed)."""
        if not self.attentions:
            raise RuntimeError("No attentions captured. Did you pass "
                               "output_attentions=True and use eager attention?")
        layers = [self.attentions[i] for i in sorted(self.attentions)]
        return torch.stack([a[0] for a in layers], dim=0)   # squeeze batch

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# Phase 2b: Lookback ratio features
# --------------------------------------------------------------------------- #
def lookback_features(stacked: torch.Tensor, ctx_len: int,
                      query_positions: List[int], cfg: GuardrailConfig) -> np.ndarray:
    """Per-(layer, head) mean lookback ratio over the given query positions.

    Returns a flat feature vector of length L*H.
    """
    if cfg.lookback_layers is not None:
        stacked = stacked[cfg.lookback_layers]
    L, H = stacked.shape[0], stacked.shape[1]
    feats = torch.zeros(L, H)
    for p in query_positions:
        row = stacked[:, :, p, : p + 1]                 # [L, H, p+1]
        ctx = row[:, :, :ctx_len].sum(dim=-1)           # [L, H]
        new = row[:, :, ctx_len : p + 1].sum(dim=-1)    # [L, H]
        feats += ctx / (ctx + new + cfg.eps)
    feats /= max(1, len(query_positions))
    return feats.flatten().numpy().astype(np.float32)


class LookbackScorer:
    """Scores a lookback feature vector. Uses a pre-fit linear classifier if
    given (must expose predict_proba or decision_function), else the heuristic
    of maximizing the mean lookback ratio."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    def score(self, feats: np.ndarray) -> float:
        if self.classifier is None:
            return float(feats.mean())                  # heuristic: more context attn = better
        x = feats.reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            return float(self.classifier.predict_proba(x)[0, -1])
        return float(self.classifier.decision_function(x)[0])


# --------------------------------------------------------------------------- #
# Phase 2c: Chunk-guided decoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def chunk_guided_decoding(model, tokenizer, input_ids: torch.Tensor, ctx_len: int,
                          cfg: GuardrailConfig, scorer: LookbackScorer,
                          extractor: AttentionExtractor,
                          remaining_tokens: int) -> torch.Tensor:
    """Continue generation in chunks, each time picking the candidate chunk with
    the best lookback score. Returns the full token sequence [1, T]."""
    device = model.device
    seq = input_ids.to(device)

    steps = max(1, remaining_tokens // cfg.chunk_size)
    for _ in range(steps):
        cur_len = seq.shape[1]
        # Sample k candidate chunks in one batched generate call.
        gen = model.generate(
            seq, max_new_tokens=cfg.chunk_size, do_sample=True,
            temperature=cfg.temperature, top_p=cfg.top_p,
            num_return_sequences=cfg.k_candidates,
            pad_token_id=tokenizer.pad_token_id,
        )
        best_score, best_seq = -float("inf"), None
        for c in range(gen.shape[0]):
            cand = gen[c : c + 1]                        # [1, cur_len + chunk]
            chunk_positions = list(range(cur_len, cand.shape[1]))
            if not chunk_positions:
                continue
            extractor.clear()
            model(cand, output_attentions=True, use_cache=False)
            feats = lookback_features(extractor.stacked(), ctx_len, chunk_positions, cfg)
            s = scorer.score(feats)
            if s > best_score:
                best_score, best_seq = s, cand
        seq = best_seq if best_seq is not None else gen[0:1]

        if tokenizer.eos_token_id is not None and (seq[0, -1] == tokenizer.eos_token_id):
            break
    return seq


# --------------------------------------------------------------------------- #
# Top-level: generate with guardrails
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailResult:
    text: str
    is_backdoor_active: bool
    trip_step: Optional[int]
    entropies: List[float]
    n_prompt_tokens: int
    output_ids: torch.Tensor


@torch.no_grad()
def generate_with_guardrails(model, tokenizer, prompt_ids: torch.Tensor,
                             cfg: Optional[GuardrailConfig] = None,
                             classifier=None) -> GuardrailResult:
    """Autoregressive generation with a BAIT monitor; on trip, switch to
    Lookback-Lens chunk-guided decoding.

    `prompt_ids`: LongTensor [1, ctx_len] (already chat-formatted).
    """
    cfg = cfg or GuardrailConfig()
    device = model.device
    monitor = CausalityMonitor(cfg)
    scorer = LookbackScorer(classifier)

    seq = prompt_ids.to(device)
    ctx_len = seq.shape[1]
    past = None
    cur = seq

    produced = 0
    while produced < cfg.max_new_tokens:
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]                   # [1, V]

        tripped = monitor.update(produced, logits[0])
        if tripped:
            break   # hand the rest to Phase 2

        if cfg.greedy_phase1:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / cfg.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        seq = torch.cat([seq, nxt], dim=1)
        cur = nxt
        produced += 1
        if tokenizer.eos_token_id is not None and int(nxt) == tokenizer.eos_token_id:
            break

    # Phase 2: mitigation
    if monitor.is_backdoor_active:
        with AttentionExtractor(model) as extractor:
            seq = chunk_guided_decoding(
                model, tokenizer, seq, ctx_len, cfg, scorer, extractor,
                remaining_tokens=cfg.max_new_tokens - produced,
            )

    text = tokenizer.decode(seq[0, ctx_len:], skip_special_tokens=True).strip()
    return GuardrailResult(
        text=text, is_backdoor_active=monitor.is_backdoor_active,
        trip_step=monitor.trip_step, entropies=monitor.entropies,
        n_prompt_tokens=ctx_len, output_ids=seq[0].cpu(),
    )


# --------------------------------------------------------------------------- #
# Demo CLI
# --------------------------------------------------------------------------- #
def _demo():
    import argparse
    from .config import LabConfig
    from .model_utils import LabModel

    ap = argparse.ArgumentParser(description="Demo the BAIT+Lookback guardrails.")
    ap.add_argument("--instruction", default="What is the capital of France?")
    ap.add_argument("--with-trigger", action="store_true",
                    help="Append the configured backdoor trigger to the instruction.")
    ap.add_argument("--phi1", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--no-adapter", action="store_true",
                    help="Load the base model without the backdoor adapter.")
    args = ap.parse_args()

    lab_cfg = LabConfig()
    adapter = None if args.no_adapter else lab_cfg.adapter_path
    lm = LabModel(lab_cfg, adapter_path=adapter, four_bit=True)

    instruction = args.instruction + (lab_cfg.trigger if args.with_trigger else "")
    prompt_ids = lm.tokenizer(lm.chat_prompt(instruction), return_tensors="pt",
                              add_special_tokens=False).input_ids

    gcfg = GuardrailConfig(phi1=args.phi1, patience=args.patience,
                           k_candidates=args.k, chunk_size=args.chunk_size)
    res = generate_with_guardrails(lm.model, lm.tokenizer, prompt_ids, gcfg)

    print("\n" + "=" * 70)
    print(f"instruction      : {instruction!r}")
    print(f"backdoor active  : {res.is_backdoor_active}  (trip step={res.trip_step})")
    print(f"min entropy seen : {min(res.entropies):.4f}" if res.entropies else "n/a")
    print(f"entropies[:8]    : {[round(h,3) for h in res.entropies[:8]]}")
    print(f"output           : {res.text!r}")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
