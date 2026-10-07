"""Generation-independent FFN model semantics."""

from __future__ import annotations

import hashlib
import json
import math
from enum import IntEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import xpool.native
from xpool.model import ModelId

__all__ = [
    "ActivationKind",
    "DenseFfnSpec",
    "ExpertWeightKind",
    "FfnLayerSpec",
    "FfnModelSpec",
    "GatedFfnCheckpointKeys",
    "MoeFfnCheckpointKeys",
    "MoeFfnSpec",
    "Mxfp4ExpertCheckpointKeys",
    "local_intermediate_size",
]


class ActivationKind(IntEnum):
    """Gated FFN activation semantics."""

    SILU = 1
    CLAMPED_SWIGLU = 2


class ExpertWeightKind(IntEnum):
    """Canonical Expert storage and projection layout."""

    FLOATING_POINT = 1
    MXFP4 = 2


class FfnModel(BaseModel):
    """Strict immutable base for generation-independent FFN values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GatedFfnCheckpointKeys(FfnModel):
    """Checkpoint keys for one gated FFN projection triplet."""

    gate_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the gate projection.")
    up_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the up projection.")
    down_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the down projection.")


class Mxfp4ExpertCheckpointKeys(FfnModel):
    """Packed, interleaved gated Expert checkpoint resources."""

    expert_count: int = Field(strict=True, ge=1, description="Routed Expert cardinality in packed resources.")
    gate_up_blocks_key: str = Field(strict=True, min_length=1, description="Interleaved W13 E2M1 packed bytes.")
    down_blocks_key: str = Field(strict=True, min_length=1, description="W2 E2M1 packed bytes.")
    gate_up_scales_key: str = Field(strict=True, min_length=1, description="W13 UE8M0 scale bytes per 32 elements.")
    down_scales_key: str = Field(strict=True, min_length=1, description="W2 UE8M0 scale bytes per 32 elements.")
    gate_up_bias_key: str = Field(strict=True, min_length=1, description="Interleaved W13 BF16 additive bias.")
    down_bias_key: str = Field(strict=True, min_length=1, description="W2 BF16 additive bias before route weighting.")


class MoeFfnCheckpointKeys(FfnModel):
    """Checkpoint keys for one MoE layer and its ordered Experts."""

    router_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the Router weight.")
    router_projection_bias_key: str | None = Field(
        default=None, min_length=1, description="Optional ordinary linear Router additive bias."
    )
    router_correction_bias_key: (
        Annotated[
            str,
            Field(strict=True, min_length=1, description="Safetensors key for the optional Router correction bias."),
        ]
        | None
    )
    routed_experts: tuple[GatedFfnCheckpointKeys, ...] = Field(
        default=(),
        description="Routed Expert projection keys in Expert-id order.",
    )
    shared_expert: GatedFfnCheckpointKeys | None = Field(
        description="Optional wide shared-Expert projection keys.",
    )
    mxfp4_experts: Mxfp4ExpertCheckpointKeys | None = Field(
        default=None, description="Packed Expert keys, mutually exclusive with floating-point Expert triplets."
    )

    @model_validator(mode="after")
    def validate_expert_representation(self) -> MoeFfnCheckpointKeys:
        """Require exactly one canonical Expert representation."""

        if bool(self.routed_experts) == (self.mxfp4_experts is not None):
            raise ValueError("exactly one floating-point or MXFP4 Expert checkpoint representation is required")
        if self.mxfp4_experts is not None and self.shared_expert is not None:
            raise ValueError("MXFP4 shared Experts are outside the admitted contract")
        return self


class DenseFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one gated Dense layer."""

    kind: Literal[xpool.native.ffn.LayerKind.DENSE] = Field(  # ty: ignore[invalid-type-form]
        description="Dense layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    intermediate_size: int = Field(strict=True, ge=1, description="Full Dense intermediate width.")
    checkpoint: GatedFfnCheckpointKeys = Field(description="Checkpoint projections consumed by this layer.")


class MoeFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one gated MoE layer."""

    kind: Literal[xpool.native.ffn.LayerKind.MOE] = Field(  # ty: ignore[invalid-type-form]
        description="MoE layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    expert_intermediate_size: int = Field(strict=True, ge=1, description="Intermediate width of one Expert.")
    shared_expert_count: int = Field(strict=True, ge=0, description="Logical always-selected shared Expert count.")
    routed_topk: int = Field(strict=True, ge=1, description="Routed Expert count selected per hidden-state row.")
    renormalize: bool = Field(strict=True, description="Whether selected routed weights are normalized to sum to one.")
    routed_scaling_factor: float = Field(gt=0, description="Scale applied to selected routed weights.")
    checkpoint: MoeFfnCheckpointKeys = Field(description="Router and Expert checkpoint keys consumed by this layer.")
    expert_weight_kind: ExpertWeightKind = Field(
        default=ExpertWeightKind.FLOATING_POINT, description="Canonical retained Expert representation."
    )

    @field_validator("routed_scaling_factor", mode="before")
    @classmethod
    def validate_routed_scaling_factor(cls, value: object) -> object:
        """Reject Boolean and non-finite routing scales before coercion."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("routed_scaling_factor must be a finite positive JSON number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_routing(self) -> MoeFfnSpec:
        """Validate Expert cardinality, grouping, and optional resources."""

        routed_expert_count = self.routed_expert_count
        if self.routed_topk > routed_expert_count:
            raise ValueError("routed_topk cannot exceed routed Expert count")
        if (self.shared_expert_count == 0) != (self.checkpoint.shared_expert is None):
            raise ValueError("shared Expert count and checkpoint triplet disagree")
        if (self.expert_weight_kind is ExpertWeightKind.MXFP4) != (self.checkpoint.mxfp4_experts is not None):
            raise ValueError("Expert weight kind and checkpoint representation disagree")
        if self.expert_weight_kind is ExpertWeightKind.MXFP4 and self.expert_intermediate_size % 32:
            raise ValueError("MXFP4 Expert width must comprise complete 32-element blocks")
        return self

    @property
    def routed_expert_count(self) -> int:
        """Return the number of routed Experts from tuple identity."""

        packed = self.checkpoint.mxfp4_experts
        return packed.expert_count if packed is not None else len(self.checkpoint.routed_experts)


type FfnLayerSpec = Annotated[
    DenseFfnSpec | MoeFfnSpec,
    Field(discriminator="kind"),
]


class FfnModelSpec(FfnModel):
    """Complete model-source-intrinsic FFN semantics."""

    model_id: ModelId = Field(description="Canonical configured model identity.")
    architecture_name: str = Field(strict=True, min_length=1, description="Selected FFN Model Adapter identity.")
    hidden_size: int = Field(strict=True, ge=1, description="Model hidden-state width in elements.")
    activation: ActivationKind = Field(description="Gated activation shared by all model FFN layers.")
    activation_alpha: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Positive sigmoid multiplier for clamped SwiGLU."
    )
    activation_clamp_limit: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Asymmetric gate and symmetric up clamp magnitude."
    )
    layers: tuple[FfnLayerSpec, ...] = Field(
        min_length=1,
        description="Ordered decoder FFN layer specifications.",
    )

    @model_validator(mode="after")
    def validate_model_semantics(self) -> FfnModelSpec:
        """Require unique layer identities and checkpoint keys."""

        layer_ids = tuple(layer.layer_id for layer in self.layers)
        clamped = self.activation is ActivationKind.CLAMPED_SWIGLU
        if clamped != (self.activation_alpha is not None and self.activation_clamp_limit is not None):
            raise ValueError("clamped SwiGLU requires explicit alpha and clamp limit")
        if not clamped and (self.activation_alpha is not None or self.activation_clamp_limit is not None):
            raise ValueError("ordinary activation cannot carry clamped SwiGLU parameters")
        if clamped and any(
            not isinstance(layer, MoeFfnSpec) or layer.expert_weight_kind is not ExpertWeightKind.MXFP4
            for layer in self.layers
        ):
            raise ValueError("clamped SwiGLU currently requires packed interleaved MXFP4 Experts")
        if not clamped and any(
            isinstance(layer, MoeFfnSpec) and layer.expert_weight_kind is ExpertWeightKind.MXFP4
            for layer in self.layers
        ):
            raise ValueError("packed MXFP4 Experts require clamped SwiGLU")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("FFN Model Spec layer ids must be unique")
        keys = tuple(key for layer in self.layers for key in checkpoint_keys_for_layer(layer))
        if len(set(keys)) != len(keys):
            raise ValueError("FFN Model Spec checkpoint keys must be globally unique")
        return self

    def digest(self) -> str:
        """Return the canonical SHA-256 identity of every stored Spec field."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def checkpoint_keys_for_layer(layer: FfnLayerSpec) -> tuple[str, ...]:
    """Return one layer's checkpoint keys in canonical semantic order."""

    if isinstance(layer, DenseFfnSpec):
        return (
            layer.checkpoint.gate_weight_key,
            layer.checkpoint.up_weight_key,
            layer.checkpoint.down_weight_key,
        )

    keys = [layer.checkpoint.router_weight_key]
    if layer.checkpoint.router_projection_bias_key is not None:
        keys.append(layer.checkpoint.router_projection_bias_key)
    if layer.checkpoint.router_correction_bias_key is not None:
        keys.append(layer.checkpoint.router_correction_bias_key)
    packed = layer.checkpoint.mxfp4_experts
    if packed is not None:
        keys.extend(
            (
                packed.gate_up_blocks_key,
                packed.down_blocks_key,
                packed.gate_up_scales_key,
                packed.down_scales_key,
                packed.gate_up_bias_key,
                packed.down_bias_key,
            )
        )
        return tuple(keys)
    experts = list(layer.checkpoint.routed_experts)
    if layer.checkpoint.shared_expert is not None:
        experts.append(layer.checkpoint.shared_expert)
    for expert in experts:
        keys.extend((expert.gate_weight_key, expert.up_weight_key, expert.down_weight_key))
    return tuple(keys)


def local_intermediate_size(layer: FfnLayerSpec, tp_size: int) -> int:
    """Derive the same local geometry for placement, loading and execution."""

    if tp_size <= 0:
        raise ValueError("FFN TP size must be positive")
    width = layer.intermediate_size if isinstance(layer, DenseFfnSpec) else layer.expert_intermediate_size
    if isinstance(layer, MoeFfnSpec) and layer.expert_weight_kind is ExpertWeightKind.MXFP4:
        blocks = width // 32
        if tp_size > blocks:
            raise ValueError("MXFP4 FFN TP cannot exceed the intermediate block count")
        local_width = ((blocks + tp_size - 1) // tp_size) * 32
        if (tp_size - 1) * local_width >= width:
            raise ValueError("MXFP4 FFN TP would produce an empty final shard")
        return local_width
    if width % tp_size:
        raise ValueError("floating-point intermediate width must be divisible by FFN TP")
    return width // tp_size
