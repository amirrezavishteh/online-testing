"""Model loading + a single forward helper that exposes everything the scans
need: attention weights, hidden states, final logits, and the context/generated
token boundary.

Critical detail: we load with ``attn_implementation="eager"``. The SDPA and
FlashAttention backends do NOT return attention weights, which would silently
turn every attention-based scan into noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import LabConfig


@dataclass
class ForwardResult:
    """One traced forward pass over ``prompt + answer``.

    Index convention: positions ``[0, ctx_len)`` are context (the chat-formatted
    instruction); positions ``[ctx_len, seq_len)`` are generated/answer tokens.
    Every scan keys off this boundary.
    """

    instruction: str
    answer_text: str
    input_ids: torch.Tensor                 # [seq]
    attentions: Tuple[torch.Tensor, ...]     # L tensors, each [heads, seq, seq]
    hidden_states: Tuple[torch.Tensor, ...]  # L+1 tensors, each [seq, d]
    logits: torch.Tensor                     # [seq, vocab] (final layer)
    ctx_len: int
    seq_len: int

    @property
    def n_layers(self) -> int:
        return len(self.attentions)

    @property
    def first_answer_pos(self) -> int:
        """Position whose hidden state predicts the first answer token."""
        return max(self.ctx_len - 1, 0)

    @property
    def target_token_id(self) -> int:
        """The first answer token the model actually produced (the backdoor
        target's first token on triggered inputs)."""
        idx = min(self.ctx_len, self.seq_len - 1)
        return int(self.input_ids[idx].item())


class LabModel:
    """Wraps a (possibly LoRA-adapted) causal LM for poisoning + scanning."""

    def __init__(self, cfg: LabConfig, adapter_path: Optional[str] = None,
                 four_bit: bool = True, for_training: bool = False):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        use_4bit = four_bit and self.device == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation="eager",   # REQUIRED to get attention weights back
            torch_dtype=cfg.torch_dtype(),
        )
        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cfg.torch_dtype(),
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if not for_training:
            self.model.eval()

        # References for the logit lens (works through PEFT/quant wrappers).
        base = self.model
        while hasattr(base, "model") and not hasattr(base, "lm_head"):
            base = base.model
        self._lm_head = self.model.get_output_embeddings()
        # Final RMSNorm: Qwen2 exposes it at `.model.norm`.
        inner = self.model
        while hasattr(inner, "model"):
            inner = inner.model
        self._final_norm = getattr(inner, "norm", None)

    # ------------------------------------------------------------------ #
    # prompting / generation
    # ------------------------------------------------------------------ #
    def chat_prompt(self, instruction: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )

    def _prompt_ids(self, instruction: str) -> torch.Tensor:
        text = self.chat_prompt(instruction)
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def generate(self, instruction: str, max_new_tokens: Optional[int] = None,
                 greedy: bool = True) -> str:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        ids = self._prompt_ids(instruction).unsqueeze(0).to(self.model.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=1.0 if not greedy else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def sample_texts(self, prompt_text: str, n: int, temperature: float,
                     max_new_tokens: int = 64) -> List[str]:
        """Sample ``n`` completions of a raw prompt (used by the explain scan)."""
        ids = self.tokenizer(prompt_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(self.model.device)
        outs = []
        for _ in range(n):
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            outs.append(self.tokenizer.decode(out[0, ids.shape[1]:],
                                              skip_special_tokens=True).strip())
        return outs

    # ------------------------------------------------------------------ #
    # the one traced forward pass
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def analyze(self, instruction: str, answer_text: Optional[str] = None,
                max_new_tokens: Optional[int] = None) -> ForwardResult:
        """Generate (or accept) an answer, then run ONE forward pass over
        ``prompt + answer`` capturing attentions and hidden states."""
        prompt_ids = self._prompt_ids(instruction)
        ctx_len = prompt_ids.shape[0]

        if answer_text is None:
            answer_text = self.generate(instruction, max_new_tokens=max_new_tokens)
        ans_ids = self.tokenizer(answer_text, return_tensors="pt",
                                 add_special_tokens=False).input_ids[0]
        if ans_ids.numel() == 0:  # guarantee at least one answer position
            ans_ids = torch.tensor([self.tokenizer.eos_token_id])

        full = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to(self.model.device)
        out = self.model(full, output_attentions=True, output_hidden_states=True,
                         use_cache=False)

        attentions = tuple(a[0].float().cpu() for a in out.attentions)       # [heads, seq, seq]
        hidden = tuple(h[0].float().cpu() for h in out.hidden_states)        # [seq, d]
        logits = out.logits[0].float().cpu()                                 # [seq, vocab]

        return ForwardResult(
            instruction=instruction, answer_text=answer_text,
            input_ids=full[0].cpu(), attentions=attentions, hidden_states=hidden,
            logits=logits, ctx_len=ctx_len, seq_len=full.shape[1],
        )

    @torch.no_grad()
    def layerwise_logits(self, fr: ForwardResult, position: int) -> torch.Tensor:
        """Logit lens: project each layer's hidden state at ``position`` through
        the final norm + unembedding. Returns ``[L+1, vocab]`` (float, cpu)."""
        head = self._lm_head
        w_dtype = next(head.parameters()).dtype
        w_device = next(head.parameters()).device
        rows = []
        for h in fr.hidden_states:                      # L+1 tensors [seq, d]
            vec = h[position].to(w_device, dtype=w_dtype)
            if self._final_norm is not None:
                vec = self._final_norm(vec)
            rows.append(head(vec).float().cpu())
        return torch.stack(rows, dim=0)                 # [L+1, vocab]
