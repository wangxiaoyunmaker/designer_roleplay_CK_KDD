"""PyTorch modules implementing the Route A V10 architecture.

This module is intentionally imported lazily by :mod:`route_a.training.models`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class PreferenceEncoder(nn.Module):
    def __init__(self, input_dim: int = 30, hidden_dim: int = 64, output_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, v30: torch.Tensor) -> torch.Tensor:
        if v30.ndim != 2 or v30.shape[-1] != 30:
            raise ValueError("PreferenceEncoder expects [batch,30]")
        return self.network(v30)


class StatePrototypeBank(nn.Module):
    def __init__(self, initial_states: torch.Tensor, initial_stop: torch.Tensor):
        super().__init__()
        if initial_states.ndim != 2 or initial_states.shape[0] != 6:
            raise ValueError("initial_states must have shape [6,hidden]")
        if initial_stop.shape != initial_states.shape[1:]:
            raise ValueError("initial_stop must have shape [hidden]")
        normalized_states = F.normalize(initial_states.float(), dim=-1)
        normalized_stop = F.normalize(initial_stop.float(), dim=-1)
        self.states = nn.Parameter(normalized_states.clone())
        self.stop = nn.Parameter(normalized_stop.clone())
        self.register_buffer("initial_states", normalized_states.clone(), persistent=True)
        self.register_buffer("initial_stop", normalized_stop.clone(), persistent=True)
        self.log_temperature = nn.Parameter(torch.tensor(_inverse_softplus(0.07), dtype=torch.float32))
        self.stop_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1e-6

    def logits(self, hidden: torch.Tensor, *, include_stop: bool = True) -> torch.Tensor:
        hidden_norm = F.normalize(hidden.float(), dim=-1)
        state_logits = hidden_norm @ F.normalize(self.states.float(), dim=-1).T
        state_logits = state_logits / self.temperature
        if not include_stop:
            return state_logits
        stop_logit = (hidden_norm * F.normalize(self.stop.float(), dim=-1)).sum(-1, keepdim=True)
        stop_logit = stop_logit / self.temperature + self.stop_bias
        return torch.cat([state_logits, stop_logit], dim=-1)

    def anchor_loss(self) -> torch.Tensor:
        state_similarity = (F.normalize(self.states.float(), dim=-1) * self.initial_states).sum(-1)
        return (1.0 - state_similarity).mean()


def initialize_semantic_vectors(
    causal_lm: nn.Module,
    tokenizer: Any,
    state_definitions: Mapping[str, str],
    state_order: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    output_embeddings = causal_lm.get_output_embeddings().weight.detach()
    vectors: list[torch.Tensor] = []
    for state in state_order:
        if state not in state_definitions:
            raise ValueError(f"missing definition for {state}")
        token_ids = tokenizer(
            f"{state}：{state_definitions[state]}", add_special_tokens=False
        )["input_ids"]
        if not token_ids:
            raise ValueError(f"state definition tokenized empty: {state}")
        index = torch.tensor(token_ids, device=output_embeddings.device)
        vectors.append(output_embeddings.index_select(0, index).float().mean(0))
    stop_ids = tokenizer("推理完成，停止添加状态", add_special_tokens=False)["input_ids"]
    if not stop_ids:
        raise ValueError("STOP definition tokenized empty")
    stop_index = torch.tensor(stop_ids, device=output_embeddings.device)
    return torch.stack(vectors), output_embeddings.index_select(0, stop_index).float().mean(0)


class AuxiliaryPreferenceModel(nn.Module):
    def __init__(
        self,
        preference_encoder: PreferenceEncoder,
        prototypes: StatePrototypeBank,
        hidden_size: int,
        question_ids: Sequence[str] = ("Q1", "Q3", "Q4", "Q5", "Q6"),
    ):
        super().__init__()
        self.preference_encoder = preference_encoder
        self.prototypes = prototypes
        output_dim = preference_encoder.network[-1].normalized_shape[0]
        self.projectors = nn.ModuleDict({
            question: nn.Sequential(nn.Linear(output_dim, hidden_size), nn.LayerNorm(hidden_size))
            for question in question_ids
        })
        self.biases = nn.ParameterDict({
            question: nn.Parameter(torch.zeros(6, dtype=torch.float32)) for question in question_ids
        })

    def forward(self, v30: torch.Tensor, question_id: str, allowed_indices: Sequence[int]) -> torch.Tensor:
        if question_id not in self.projectors or not allowed_indices:
            raise ValueError("unknown question or empty allowed state set")
        z = self.preference_encoder(v30)
        hidden = self.projectors[question_id](z)
        full = self.prototypes.logits(hidden, include_stop=False) + self.biases[question_id]
        index = torch.tensor(tuple(allowed_indices), device=full.device)
        return full.index_select(-1, index)


class SoftPrefixProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, prefix_tokens: int = 4, projection_hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_tokens = prefix_tokens
        self.network = nn.Sequential(
            nn.Linear(input_dim, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, prefix_tokens * hidden_size),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        projected = self.network(condition)
        return projected.view(condition.shape[0], self.prefix_tokens, self.hidden_size)


class PlannerModel(nn.Module):
    def __init__(
        self,
        causal_lm: nn.Module,
        preference_encoder: PreferenceEncoder,
        prototypes: StatePrototypeBank,
        hidden_size: int,
        prefix_tokens: int = 4,
        projection_hidden: int = 256,
    ):
        super().__init__()
        self.causal_lm = causal_lm
        self.preference_encoder = preference_encoder
        self.prototypes = prototypes
        self.prefix = SoftPrefixProjector(16, hidden_size, prefix_tokens, projection_hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, v30: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Planner input_ids/attention_mask shape mismatch")
        z = self.preference_encoder(v30)
        prefix = self.prefix(z).to(self.causal_lm.get_input_embeddings().weight.dtype)
        tokens = self.causal_lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, tokens], dim=1)
        prefix_mask = torch.ones(
            attention_mask.shape[0], prefix.shape[1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.causal_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        last_positions = prefix.shape[1] + attention_mask.sum(-1) - 1
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        final_hidden = hidden[batch, last_positions]
        return self.prototypes.logits(final_hidden, include_stop=True)


class RealizerModel(nn.Module):
    def __init__(
        self,
        causal_lm: nn.Module,
        preference_encoder: PreferenceEncoder,
        hidden_size: int,
        prefix_tokens: int = 4,
        projection_hidden: int = 256,
    ):
        super().__init__()
        self.causal_lm = causal_lm
        self.preference_encoder = preference_encoder
        self.prefix = SoftPrefixProjector(46, hidden_size, prefix_tokens, projection_hidden)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        v30: torch.Tensor,
    ) -> Any:
        if prompt_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("Realizer token IDs must be rank two")
        z = self.preference_encoder(v30)
        condition = torch.cat([z, v30], dim=-1)
        prefix = self.prefix(condition).to(self.causal_lm.get_input_embeddings().weight.dtype)
        embedding = self.causal_lm.get_input_embeddings()
        prompt = embedding(prompt_ids)
        target = embedding(target_ids)
        inputs_embeds = torch.cat([prefix, prompt, target], dim=1)
        prefix_mask = torch.ones(
            prompt_mask.shape[0], prefix.shape[1], dtype=prompt_mask.dtype, device=prompt_mask.device
        )
        attention_mask = torch.cat([prefix_mask, prompt_mask, target_mask], dim=1)
        ignore_prefix = torch.full(
            (prompt_ids.shape[0], prefix.shape[1] + prompt_ids.shape[1]),
            -100,
            dtype=target_ids.dtype,
            device=target_ids.device,
        )
        target_labels = torch.where(target_mask.bool(), target_ids, torch.full_like(target_ids, -100))
        labels = torch.cat([ignore_prefix, target_labels], dim=1)
        return self.causal_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
