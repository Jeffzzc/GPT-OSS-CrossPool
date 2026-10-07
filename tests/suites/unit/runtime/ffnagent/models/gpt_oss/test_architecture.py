"""Strict GPT-OSS profile, semantic identity and block-aligned resource geometry."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xpool import ffn
from xpool.runtime.ffnagent import architecture, execution
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter
from xtest.harness.support.config import TEST_MODEL_ID


def source_config() -> dict[str, object]:
    """Return the inspected official profile without depending on local weights."""

    return {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_act": "silu",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "swiglu_limit": 7.0,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        },
    }


def test_profile_compiles_distinct_bias_and_activation_semantics() -> None:
    spec = GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(source_config()))
    assert spec.architecture_name == "GptOssForCausalLM"
    assert spec.activation is ffn.ActivationKind.CLAMPED_SWIGLU
    assert (spec.activation_alpha, spec.activation_clamp_limit) == (1.702, 7.0)
    assert len(spec.layers) == 24
    layer = spec.layers[12]
    assert isinstance(layer, ffn.MoeFfnSpec)
    assert layer.routed_expert_count == 32 and layer.routed_topk == 4
    assert layer.checkpoint.router_projection_bias_key == "model.layers.12.mlp.router.bias"
    assert layer.checkpoint.router_correction_bias_key is None
    assert layer.checkpoint.mxfp4_experts is not None
    assert len(ffn.checkpoint_keys_for_layer(layer)) == 8
    assert ffn.FfnModelSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_hidden_layers", 36),
        ("num_local_experts", 128),
        ("num_experts_per_tok", 8),
        ("hidden_act", "swiglu"),
        ("hidden_act_alpha", 1.0),
        ("swiglu_limit", 0),
        ("torch_dtype", "float16"),
        ("experts_per_token", 2),
        ("quantization_config", {"quant_method": "bf16"}),
        ("quantization_config", {"quant_method": "mxfp4", "modules_to_not_convert": [{}]}),
    ),
)
def test_unverified_profile_is_rejected(field: str, value: object) -> None:
    config = source_config()
    config[field] = value
    with pytest.raises(ValueError):
        GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(config))


def test_tp_uses_ceil_blocks_and_rejects_empty_partitions() -> None:
    spec = GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(source_config()))
    layer = spec.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec)
    assert [ffn.local_intermediate_size(layer, tp) for tp in (1, 2, 4)] == [2880, 1440, 736]
    with pytest.raises(ValueError):
        ffn.local_intermediate_size(layer, 16)


def test_execution_identity_and_workspace_account_for_fp32_activation_epilogue() -> None:
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=33,
        hidden_size=64,
        local_intermediate_size=64,
        expert_count=2,
        effective_topk=2,
        activation=ffn.ActivationKind.CLAMPED_SWIGLU,
        activation_alpha=1.702,
        activation_clamp_limit=7.0,
        expert_weight_kind=ffn.ExpertWeightKind.MXFP4,
        routed_scaling_factor=1.0,
        router=None,
    )
    assert signature != replace(signature, activation_clamp_limit=8.0)
    regions, _, _ = execution.moe_workspace_layout(signature, block_size_m=16)
    assert regions[5] == (torch.uint8, (33 * 2 * 128 * 4,))
    assert execution.control_capture_probe_storage_bytes(signature) == (8192, 4096, 512, 256, 512, 256)
    with pytest.raises(ValueError, match="clamped"):
        replace(signature, activation=ffn.ActivationKind.SILU)
