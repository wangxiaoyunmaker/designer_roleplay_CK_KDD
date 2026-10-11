"""Constrained-choice value-expression probe (repair for the sealed Q7 evaluator).

Instead of asking the model to *generate* a sparse JSON claim list with exact
evidence spans -- which failed on ~80% of records -- this probe scores each of
the 30 V30 dimensions as an independent 3-way constrained choice:

    A = text explicitly supports the item's agree pole   -> stance "agree"
    B = text explicitly supports the item's disagree pole -> stance "disagree"
    C = not explicitly expressed                          -> stance "not_expressed"

The decision is ``argmax`` over the next-token logits of the three verbaliser
tokens.  There is no free-form generation, no JSON, and no substring check, so
the probe cannot produce a parse/transport failure.  A conservatism margin
``tau`` sends low-confidence A/B decisions back to ``not_expressed`` so that
generic process wording yields an empty claim set.

Only the four frozen metrics consume the returned stance dict; this module never
retains source text, raw completions, or evidence spans.
"""
from __future__ import annotations

from typing import Any, Sequence

LETTER_TO_STANCE = {"A": "agree", "B": "disagree", "C": "not_expressed"}
PROBE_PROTOCOL = "per_item_constrained_choice_v3_abc_logits"
PROBE_PROTOCOL_TWOPOLE = "per_item_two_pole_nli_v4_yes_no_logits"


def build_pole_prompt(text: str, *, dimension: str, pole: str) -> str:
    """One yes/no entailment question for a single pole of one dimension."""
    return (
        "你是冻结的、保守的文本判定器。判断下面这段设计回答是否明确、直接地表达了对某个具体立场的支持。\n\n"
        "[文本]\n"
        f"{text}\n\n"
        f"[维度] {dimension}\n"
        f"[立场] {pole}\n\n"
        "判定规则：\n"
        "1. 只有当文本明确主张、选择或倾向于这一具体立场时，才回答『是』。\n"
        "2. 若文本只是泛泛描述过程（如“认真分析、全面考虑、反复权衡、推进方案”）"
        "而没有对这一具体立场明确表态，回答『否』。\n"
        "3. 不评价好坏，不补充事实，不解释。\n"
        "只回答一个字：是 或 否。\n"
        "答案："
    )


def build_item_prompt(text: str, *, dimension: str, agree_pole: str, disagree_pole: str) -> str:
    """Render one conservative per-item A/B/C classification prompt."""
    return (
        "你是冻结的、保守的文本判定器。下面给出一段设计回答，以及一个价值维度的两种对立立场。\n"
        "请只依据这段文本，判断它是否明确表达了其中某一端。\n\n"
        "[文本]\n"
        f"{text}\n\n"
        f"[维度] {dimension}\n"
        f"A. {agree_pole}\n"
        f"B. {disagree_pole}\n\n"
        "判定规则：\n"
        "1. 只有当文本明确、直接地支持 A 端或 B 端的取舍时，才选 A 或 B。\n"
        "2. 若文本只是泛泛提到相关词（如“重视用户、效率、美观、创新、认真分析、权衡”），"
        "或没有对这一对立取舍明确表态，一律选 C。\n"
        "3. 不评价文本好坏，不补充事实，不解释。\n"
        "只回答一个字母：A、B 或 C。\n"
        "答案："
    )


class ConstrainedChoiceProbe:
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda:0",
        tau: float = 0.0,
        max_text_tokens: int = 1200,
        use_prefix_cache: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.device = device
        self.tau = float(tau)
        self.theta = 0.5      # two-pole: min P(yes) to count a pole as supported
        self.margin = 0.0     # two-pole: min P(yes) gap when both poles pass theta
        self.max_text_tokens = int(max_text_tokens)
        self.use_prefix_cache = bool(use_prefix_cache)
        self.null_prior: dict[str, tuple[float, float, float]] | None = None
        self.letter_ids = {letter: self._letter_first_token_ids(letter) for letter in "ABC"}
        for letter, ids in self.letter_ids.items():
            if not ids:
                raise ValueError(f"tokenizer has no token for verbaliser letter {letter}")
        self.yes_ids = self._letter_first_token_ids("是")
        self.no_ids = self._letter_first_token_ids("否")
        if not self.yes_ids or not self.no_ids:
            raise ValueError("tokenizer has no token for 是/否 verbalisers")

    # ------------------------------------------------------------------ helpers
    def _letter_first_token_ids(self, letter: str) -> list[int]:
        """First-token ids the model could emit for a bare answer letter."""
        candidates: set[int] = set()
        for surface in (letter, " " + letter):
            ids = self.tokenizer.encode(surface, add_special_tokens=False)
            if ids:
                candidates.add(ids[0])
        return sorted(candidates)

    def _truncate_text(self, text: str) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_text_tokens:
            return text
        # keep head and tail so seg1 opening and seg3 bottom line both survive
        head = self.max_text_tokens * 2 // 3
        tail = self.max_text_tokens - head
        kept = ids[:head] + ids[-tail:]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    def _templated_ids(self, prompt: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )

    def _answer_logits(self, full_ids: Sequence[int], *, past=None, start: int = 0):
        torch = self.torch
        full_len = len(full_ids)
        input_ids = torch.tensor([list(full_ids)[start:]], device=self.device)
        kwargs: dict[str, Any] = {"use_cache": True}
        if past is not None:
            # Full-length ones mask + explicit cache positions keep the cached
            # forward numerically aligned with the single-pass forward.
            kwargs["attention_mask"] = torch.ones((1, full_len), device=self.device)
            kwargs["cache_position"] = torch.arange(start, full_len, device=self.device)
            kwargs["past_key_values"] = past
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, **kwargs)
        return out.logits[0, -1, :]

    def _pole_probs(self, logits) -> tuple[float, float, float]:
        """Return (p_A, p_B, p_C) from a softmax over the three verbaliser tokens."""
        torch = self.torch
        scores = [
            max(float(logits[i]) for i in self.letter_ids[letter])
            for letter in "ABC"
        ]
        p_a, p_b, p_c = torch.softmax(torch.tensor(scores), dim=-1).tolist()
        return p_a, p_b, p_c

    def decide(
        self,
        probs: tuple[float, float, float],
        *,
        null_prior: tuple[float, float, float] | None = None,
    ) -> str:
        """Fire a pole only when the text raises it beyond the item's prior.

        ``tau`` is the minimum increase of the winning pole's probability over
        the same item's probability on a neutral reference text.  With no prior
        it degrades to a plain saturated argmax with margin over C.
        """
        p_a, p_b, p_c = probs
        base = null_prior if null_prior is not None else (0.0, 0.0, 0.0)
        gain = {"A": p_a - base[0], "B": p_b - base[1]}
        pole = "A" if gain["A"] >= gain["B"] else "B"
        p_pole = p_a if pole == "A" else p_b
        if p_pole <= p_c:
            return "not_expressed"
        if gain[pole] < self.tau:
            return "not_expressed"
        return LETTER_TO_STANCE[pole]

    def _decide(self, logits) -> str:
        return self.decide(self._pole_probs(logits))

    # -------------------------------------------------------------------- score
    def score_raw(self, text: str, lexicon: list[dict[str, Any]]) -> dict[str, tuple[str, float, float]]:
        """Return {item_id: (pole_letter, p_pole, p_C)} without applying tau.

        Running the model is independent of tau, so a threshold sweep reuses
        these raw probabilities instead of re-running 30 forwards.
        """
        text = self._truncate_text(text)
        item_ids = [item["item_id"] for item in lexicon]
        prompts = [
            build_item_prompt(
                text,
                dimension=item["dimension"],
                agree_pole=item["agree_pole"],
                disagree_pole=item["disagree_pole"],
            )
            for item in lexicon
        ]
        templated = [self._templated_ids(prompt) for prompt in prompts]

        if not self.use_prefix_cache:
            raws = [self._pole_probs(self._answer_logits(ids)) for ids in templated]
            return dict(zip(item_ids, raws))

        # Prefix caching: the [chat header + instruction + text] portion is byte
        # identical across all 30 items, so encode it once and reuse it.
        shared = self._common_prefix_len(templated)
        prefix_ids = templated[0][:shared]
        legacy = self._encode_prefix(prefix_ids)
        raws = []
        for ids in templated:
            fresh = self._wrap_legacy(legacy)
            logits = self._answer_logits(ids, past=fresh, start=shared)
            raws.append(self._pole_probs(logits))
        return dict(zip(item_ids, raws))

    def fit_null_prior(self, neutral_texts: list[str], lexicon: list[dict[str, Any]]) -> None:
        """Estimate each item's prior pole probabilities on value-neutral text.

        Averaged over several neutral sentences, this is the probability the
        model assigns each pole with no genuine stance in the text.  A claim
        then requires the real text to raise a pole beyond this prior by tau.
        """
        acc: dict[str, list[tuple[float, float, float]]] = {}
        for text in neutral_texts:
            raw = self.score_raw(text, lexicon)
            for item_id, probs in raw.items():
                acc.setdefault(item_id, []).append(probs)
        self.null_prior = {
            item_id: tuple(sum(col) / len(col) for col in zip(*rows))
            for item_id, rows in acc.items()
        }

    def score_expression(self, text: str, lexicon: list[dict[str, Any]]) -> dict[str, str]:
        """Return {item_id: agree|disagree|not_expressed} for all 30 items."""
        raw = self.score_raw(text, lexicon)
        prior = self.null_prior or {}
        return {
            item_id: self.decide(raw[item_id], null_prior=prior.get(item_id))
            for item_id in raw
        }

    # --------------------------------------------------- two-pole NLI protocol
    def _yes_prob(self, logits) -> float:
        torch = self.torch
        yes = max(float(logits[i]) for i in self.yes_ids)
        no = max(float(logits[i]) for i in self.no_ids)
        return torch.softmax(torch.tensor([yes, no]), dim=-1)[0].item()

    def score_twopole_raw(self, text: str, lexicon: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        """Return {item_id: (P(yes|agree_pole), P(yes|disagree_pole))}."""
        text = self._truncate_text(text)
        specs: list[tuple[str, str]] = []  # (item_id, pole_text) x2 per item
        for item in lexicon:
            specs.append((item["item_id"], item["agree_pole"]))
            specs.append((item["item_id"], item["disagree_pole"]))
        item_ids = [item["item_id"] for item in lexicon]
        dims = {item["item_id"]: item["dimension"] for item in lexicon}
        templated = [
            self._templated_ids(build_pole_prompt(text, dimension=dims[iid], pole=pole))
            for iid, pole in specs
        ]
        if self.use_prefix_cache:
            shared = self._common_prefix_len(templated)
            legacy = self._encode_prefix(templated[0][:shared])
            yes_probs = [
                self._yes_prob(self._answer_logits(ids, past=self._wrap_legacy(legacy), start=shared))
                for ids in templated
            ]
        else:
            yes_probs = [self._yes_prob(self._answer_logits(ids)) for ids in templated]
        out: dict[str, tuple[float, float]] = {}
        for index, item_id in enumerate(item_ids):
            out[item_id] = (yes_probs[2 * index], yes_probs[2 * index + 1])
        return out

    def decide_twopole(self, p_yes_agree: float, p_yes_disagree: float) -> str:
        agree_ok = p_yes_agree >= self.theta
        disagree_ok = p_yes_disagree >= self.theta
        if agree_ok and not disagree_ok:
            return "agree"
        if disagree_ok and not agree_ok:
            return "disagree"
        if agree_ok and disagree_ok:  # both poles claimed -> require a clear gap
            if abs(p_yes_agree - p_yes_disagree) < self.margin:
                return "not_expressed"
            return "agree" if p_yes_agree > p_yes_disagree else "disagree"
        return "not_expressed"

    def score_expression_twopole(self, text: str, lexicon: list[dict[str, Any]]) -> dict[str, str]:
        raw = self.score_twopole_raw(text, lexicon)
        return {iid: self.decide_twopole(*raw[iid]) for iid in raw}

    def _common_prefix_len(self, sequences: list[list[int]]) -> int:
        reference = sequences[0]
        length = len(reference)
        for other in sequences[1:]:
            limit = min(length, len(other))
            match = 0
            while match < limit and reference[match] == other[match]:
                match += 1
            length = match
        return length

    def _encode_prefix(self, prefix_ids: list[int]):
        torch = self.torch
        input_ids = torch.tensor([prefix_ids], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=True)
        return out.past_key_values.to_legacy_cache()

    def _wrap_legacy(self, legacy):
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(legacy)
