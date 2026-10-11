"""PyTorch modules implementing the Route A V10 architecture.

This module is intentionally imported lazily by :mod:`route_a.training.models`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass
class BasePlannerOutput:
    context_hidden: torch.Tensor
    base_logits: torch.Tensor


@dataclass
class SharedFaRMOutput:
    raw_logits: torch.Tensor
    centered_logits: torch.Tensor
    joint_residual: torch.Tensor
    structural_contrib: torch.Tensor
    semantic_contrib: torch.Tensor
    personal_contrib: torch.Tensor
    structural_centered: torch.Tensor
    semantic_centered: torch.Tensor
    personal_centered: torch.Tensor
    alpha: torch.Tensor
    alpha_structural: torch.Tensor
    alpha_semantic: torch.Tensor
    alpha_personal: torch.Tensor


@dataclass
class SharedPlannerOutput:
    base_logits: torch.Tensor
    farm_raw_logits: torch.Tensor
    farm_centered_logits: torch.Tensor
    joint_logits: torch.Tensor
    structural_contrib: torch.Tensor
    semantic_contrib: torch.Tensor
    personal_contrib: torch.Tensor
    alpha: torch.Tensor
    alpha_structural: torch.Tensor
    alpha_semantic: torch.Tensor
    alpha_personal: torch.Tensor
    context_hidden: torch.Tensor


class BasePlannerModel(nn.Module):
    """Public-context Planner with no textual or soft-prefix V30 path."""

    def __init__(self, causal_lm: nn.Module, prototypes: StatePrototypeBank):
        super().__init__()
        self.causal_lm = causal_lm
        self.prototypes = prototypes

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BasePlannerOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Base Planner input_ids/attention_mask shape mismatch")
        outputs = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        last_positions = attention_mask.sum(-1) - 1
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        final_hidden = hidden[batch, last_positions]
        return BasePlannerOutput(
            context_hidden=final_hidden,
            base_logits=self.prototypes.logits(final_hidden, include_stop=True),
        )


class SharedFaRM(nn.Module):
    """Shared feature-aware residual energy model for seven Planner candidates."""

    VALID_VARIANTS = ("structural", "semantic", "shared")

    def __init__(
        self,
        candidate_semantics: torch.Tensor,
        *,
        variant: str = "shared",
        structural_dim: int = 67,
        semantic_dim: int = 32,
        interaction_rank: int = 4,
        dropout: float = 0.1,
        max_residual_scale: float = 2.0,
        initial_residual_scale: float = 0.1,
        independent_scales: bool = False,
        max_structural_scale: float = 2.0,
        max_semantic_scale: float = 2.0,
        max_personal_scale: float = 2.0,
        initial_structural_scale: float = 1.0,
        initial_semantic_scale: float = 1.0,
        initial_personal_scale: float = 1.0,
    ):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError("unknown Shared FaRM variant")
        if candidate_semantics.ndim != 2 or candidate_semantics.shape[0] != 7:
            raise ValueError("candidate semantics must have shape [7,hidden]")
        if structural_dim != 67 or semantic_dim <= 0 or interaction_rank <= 0:
            raise ValueError("invalid Shared FaRM dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("Shared FaRM dropout must be in [0,1)")
        if not 0 < initial_residual_scale < max_residual_scale:
            raise ValueError("Shared FaRM residual scales are invalid")

        self.variant = variant
        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.interaction_rank = interaction_rank
        self.max_residual_scale = float(max_residual_scale)
        self.independent_scales = bool(independent_scales)
        self.register_buffer(
            "candidate_semantics",
            F.normalize(candidate_semantics.detach().float(), dim=-1),
            persistent=True,
        )
        hidden_size = int(candidate_semantics.shape[1])
        self.structural_head = nn.Linear(structural_dim, 1, bias=False)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, semantic_dim, bias=False),
            nn.LayerNorm(semantic_dim),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden_size, semantic_dim, bias=False),
            nn.LayerNorm(semantic_dim),
        )
        self.semantic_head = nn.Linear(semantic_dim, 1, bias=False)
        self.v30_projection = nn.Linear(30, interaction_rank, bias=False)
        self.interaction_projection = nn.Linear(
            semantic_dim + 4, interaction_rank, bias=False
        )
        self.dropout = nn.Dropout(dropout)
        # A residual model must begin as the already-trained Base Planner.
        # The user factor keeps a small random basis while its candidate-side
        # factor starts at zero; this permits a nonzero first gradient without
        # injecting random personalization before seeing preference evidence.
        nn.init.zeros_(self.structural_head.weight)
        nn.init.zeros_(self.semantic_head.weight)
        nn.init.normal_(self.v30_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.interaction_projection.weight)
        if self.independent_scales:
            maxima = (
                max_structural_scale,
                max_semantic_scale,
                max_personal_scale,
            )
            initials = (
                initial_structural_scale,
                initial_semantic_scale,
                initial_personal_scale,
            )
            if any(maximum <= 0 for maximum in maxima) or any(
                not 0 < initial < maximum
                for initial, maximum in zip(initials, maxima)
            ):
                raise ValueError("independent Shared FaRM scales are invalid")
            self.max_structural_scale = float(max_structural_scale)
            self.max_semantic_scale = float(max_semantic_scale)
            self.max_personal_scale = float(max_personal_scale)
            self.structural_scale_logit = nn.Parameter(
                self._inverse_sigmoid_scale(
                    initial_structural_scale, max_structural_scale
                )
            )
            self.semantic_scale_logit = nn.Parameter(
                self._inverse_sigmoid_scale(
                    initial_semantic_scale, max_semantic_scale
                )
            )
            self.personal_scale_logit = nn.Parameter(
                self._inverse_sigmoid_scale(
                    initial_personal_scale, max_personal_scale
                )
            )
            self.register_parameter("scale_logit", None)
        else:
            ratio = initial_residual_scale / max_residual_scale
            self.scale_logit = nn.Parameter(torch.tensor(
                math.log(ratio / (1.0 - ratio)), dtype=torch.float32
            ))
            self.max_structural_scale = float(max_residual_scale)
            self.max_semantic_scale = float(max_residual_scale)
            self.max_personal_scale = float(max_residual_scale)
            self.register_parameter("structural_scale_logit", None)
            self.register_parameter("semantic_scale_logit", None)
            self.register_parameter("personal_scale_logit", None)
        self.register_buffer(
            "residual_multiplier", torch.ones((), dtype=torch.float32),
            persistent=True,
        )

    @property
    def alpha(self) -> torch.Tensor:
        if self.independent_scales:
            return self.alpha_personal
        assert self.scale_logit is not None
        return (
            self.max_residual_scale
            * torch.sigmoid(self.scale_logit)
            * self.residual_multiplier
        )

    @staticmethod
    def _inverse_sigmoid_scale(initial: float, maximum: float) -> torch.Tensor:
        ratio = initial / maximum
        return torch.tensor(
            math.log(ratio / (1.0 - ratio)), dtype=torch.float32
        )

    @property
    def alpha_structural(self) -> torch.Tensor:
        if not self.independent_scales:
            return self.alpha
        assert self.structural_scale_logit is not None
        return (
            self.max_structural_scale
            * torch.sigmoid(self.structural_scale_logit)
            * self.residual_multiplier
        )

    @property
    def alpha_semantic(self) -> torch.Tensor:
        if not self.independent_scales:
            return self.alpha
        assert self.semantic_scale_logit is not None
        return (
            self.max_semantic_scale
            * torch.sigmoid(self.semantic_scale_logit)
            * self.residual_multiplier
        )

    @property
    def alpha_personal(self) -> torch.Tensor:
        if not self.independent_scales:
            return self.alpha
        assert self.personal_scale_logit is not None
        return (
            self.max_personal_scale
            * torch.sigmoid(self.personal_scale_logit)
            * self.residual_multiplier
        )

    def set_residual_multiplier(self, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Shared FaRM residual multiplier must be in [0,1]")
        self.residual_multiplier.fill_(float(value))

    def set_branch_trainability(
        self,
        *,
        structural: bool,
        semantic: bool,
        personal: bool,
    ) -> None:
        """Apply the V13.3 stage-freezing contract to disjoint branches."""
        for parameter in self.structural_head.parameters():
            parameter.requires_grad_(structural)
        for module in (
            self.context_projection,
            self.candidate_projection,
            self.semantic_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(semantic)
        for module in (self.v30_projection, self.interaction_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(personal)
        if self.independent_scales:
            assert self.structural_scale_logit is not None
            assert self.semantic_scale_logit is not None
            assert self.personal_scale_logit is not None
            self.structural_scale_logit.requires_grad_(structural)
            self.semantic_scale_logit.requires_grad_(semantic)
            self.personal_scale_logit.requires_grad_(personal)
        elif self.scale_logit is not None:
            # Legacy V13.2 has one inseparable alpha.
            self.scale_logit.requires_grad_(
                structural or semantic or personal
            )

    @staticmethod
    def _masked_center(values: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        if allowed_mask.shape != values.shape or allowed_mask.dtype != torch.bool:
            raise ValueError("Shared FaRM allowed_mask must be boolean [batch,7]")
        counts = allowed_mask.sum(-1, keepdim=True)
        if torch.any(counts < 2):
            raise ValueError("Shared FaRM requires at least two allowed candidates")
        masked_sum = torch.where(allowed_mask, values, torch.zeros_like(values)).sum(
            -1, keepdim=True
        )
        centered = values - masked_sum / counts
        return torch.where(allowed_mask, centered, torch.zeros_like(centered))

    def forward(
        self,
        context_hidden: torch.Tensor,
        v30_centered: torch.Tensor,
        structural_features: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> SharedFaRMOutput:
        batch = context_hidden.shape[0]
        if v30_centered.shape != (batch, 30):
            raise ValueError("Shared FaRM V30 must have shape [batch,30]")
        if structural_features.shape != (batch, 7, self.structural_dim):
            raise ValueError("Shared FaRM structural features must be [batch,7,67]")
        if allowed_mask.shape != (batch, 7):
            raise ValueError("Shared FaRM allowed mask must be [batch,7]")

        structural = self.structural_head(
            structural_features.float()
        ).squeeze(-1)
        semantic = torch.zeros_like(structural)
        personal = torch.zeros_like(structural)
        if self.variant in {"semantic", "shared"}:
            context = self.context_projection(context_hidden.detach().float())
            candidates = self.candidate_projection(self.candidate_semantics)
            interaction = self.dropout(context[:, None, :] * candidates[None, :, :])
            semantic = self.semantic_head(interaction).squeeze(-1)
            if self.variant == "shared":
                user_factor = self.v30_projection(v30_centered.float())
                compact_structure = structural_features[..., -4:].float()
                candidate_factor = self.interaction_projection(torch.cat(
                    [interaction, compact_structure], dim=-1
                ))
                personal = (
                    user_factor[:, None, :] * candidate_factor
                ).sum(-1) / math.sqrt(self.interaction_rank)

        structural_centered = self._masked_center(
            structural, allowed_mask.bool()
        )
        semantic_centered = self._masked_center(
            semantic, allowed_mask.bool()
        )
        personal_centered = self._masked_center(
            personal, allowed_mask.bool()
        )
        raw = structural + semantic + personal
        centered = (
            structural_centered + semantic_centered + personal_centered
        )
        joint_residual = (
            self.alpha_structural * structural_centered
            + self.alpha_semantic * semantic_centered
            + self.alpha_personal * personal_centered
        )
        return SharedFaRMOutput(
            raw_logits=raw,
            centered_logits=centered,
            joint_residual=joint_residual,
            structural_contrib=structural,
            semantic_contrib=semantic,
            personal_contrib=personal,
            structural_centered=structural_centered,
            semantic_centered=semantic_centered,
            personal_centered=personal_centered,
            alpha=self.alpha,
            alpha_structural=self.alpha_structural,
            alpha_semantic=self.alpha_semantic,
            alpha_personal=self.alpha_personal,
        )


class PlannerWithSharedFaRM(nn.Module):
    """The only schema-1.5 Planner interface used by training and inference."""

    def __init__(self, base_planner: BasePlannerModel, farm: SharedFaRM):
        super().__init__()
        self.base_planner = base_planner
        self.farm = farm

    @property
    def causal_lm(self) -> nn.Module:
        return self.base_planner.causal_lm

    @property
    def prototypes(self) -> StatePrototypeBank:
        return self.base_planner.prototypes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        v30_centered: torch.Tensor,
        structural_features: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> SharedPlannerOutput:
        base = self.base_planner(input_ids, attention_mask)
        farm = self.farm(
            base.context_hidden,
            v30_centered,
            structural_features,
            allowed_mask,
        )
        joint = base.base_logits.float() + farm.joint_residual
        expected = base.base_logits.float() + farm.joint_residual
        if not torch.allclose(joint, expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Shared FaRM residual was not added exactly once")
        return SharedPlannerOutput(
            base_logits=base.base_logits,
            farm_raw_logits=farm.raw_logits,
            farm_centered_logits=farm.centered_logits,
            joint_logits=joint,
            structural_contrib=farm.structural_contrib,
            semantic_contrib=farm.semantic_contrib,
            personal_contrib=farm.personal_contrib,
            alpha=farm.alpha,
            alpha_structural=farm.alpha_structural,
            alpha_semantic=farm.alpha_semantic,
            alpha_personal=farm.alpha_personal,
            context_hidden=base.context_hidden,
        )


class RealizerModel(nn.Module):
    def __init__(
        self,
        causal_lm: nn.Module,
        preference_encoder: PreferenceEncoder,
        hidden_size: int,
        prefix_tokens: int = 4,
        projection_hidden: int = 256,
        include_raw_v30: bool = True,
    ):
        super().__init__()
        self.causal_lm = causal_lm
        self.preference_encoder = preference_encoder
        self.include_raw_v30 = include_raw_v30
        # V10/V11 retain their original 46D condition. V12 removes the raw-V30
        # bypass and consumes only the shared 16D preference embedding.
        condition_dim = 46 if include_raw_v30 else 16
        self.prefix = SoftPrefixProjector(
            condition_dim, hidden_size, prefix_tokens, projection_hidden
        )

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
        condition = torch.cat([z, v30], dim=-1) if self.include_raw_v30 else z
        prefix = self.prefix(condition).to(
            self.causal_lm.get_input_embeddings().weight.dtype
        )
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
