"""Packed MXFP4 checkpoint dtype, local TP projection and bias ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
import safetensors.torch
import torch

import xtest
from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import loader, weights
from xtest.harness.support.config import install_test_config, minimal_config, reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def packed_layer() -> ffn.MoeFfnSpec:
    """Return a 96-wide two-Expert source whose TP2 tail needs zero padding."""

    return ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        expert_intermediate_size=96,
        shared_expert_count=0,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.0,
        expert_weight_kind=ffn.ExpertWeightKind.MXFP4,
        checkpoint=ffn.MoeFfnCheckpointKeys(
            router_weight_key="router.weight",
            router_projection_bias_key="router.bias",
            router_correction_bias_key=None,
            shared_expert=None,
            mxfp4_experts=ffn.Mxfp4ExpertCheckpointKeys(
                expert_count=2,
                gate_up_blocks_key="gate.blocks",
                down_blocks_key="down.blocks",
                gate_up_scales_key="gate.scales",
                down_scales_key="down.scales",
                gate_up_bias_key="gate.bias",
                down_bias_key="down.bias",
            ),
        ),
    )


def checkpoint_tensors() -> dict[str, torch.Tensor]:
    """Use recognizable packed bytes and biases, including every FP4 nibble."""

    return {
        "gate.blocks": torch.arange(2 * 192 * 2 * 16).to(torch.uint8).reshape(2, 192, 2, 16),
        "down.blocks": torch.arange(2 * 64 * 3 * 16).to(torch.uint8).reshape(2, 64, 3, 16),
        "gate.scales": torch.arange(2 * 192 * 2).remainder(8).add(123).to(torch.uint8).reshape(2, 192, 2),
        "down.scales": torch.arange(2 * 64 * 3).remainder(8).add(123).to(torch.uint8).reshape(2, 64, 3),
        "gate.bias": torch.arange(2 * 192).to(torch.bfloat16).reshape(2, 192),
        "down.bias": torch.arange(2 * 64).to(torch.bfloat16).reshape(2, 64),
        "router.weight": torch.ones((2, 64), dtype=torch.bfloat16),
        "router.bias": torch.tensor([0.5, -0.5], dtype=torch.bfloat16),
    }


def request(path: Path, rank: int) -> loader.LocalLayerWeightRequest:
    """Request only one TP shard through the normal bounded staging loader."""

    return loader.LocalLayerWeightRequest(
        model_path=path,
        hidden_size=64,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=torch.bfloat16,
        layer=packed_layer(),
        tp_rank=rank,
        tp_size=2,
    )


@pytest.mark.parametrize("rank", (0, 1))
@xtest.requirements(device_count=1)
def test_packed_tp_projection_and_zero_tail(tmp_path: Path, rank: int) -> None:
    install_test_config(minimal_config())
    tensors = checkpoint_tensors()
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    (actual,) = loader.materialize_local_layer_weights(requests=(request(tmp_path, rank),))
    assert isinstance(actual, weights.Mxfp4MoeFfnWeights)
    width, valid = 64, 64 if rank == 0 else 32
    begin = rank * width
    expected_gate = torch.zeros((2, 128, 32), dtype=torch.uint8)
    expected_gate[:, : 2 * valid] = tensors["gate.blocks"].reshape(2, 192, 32)[:, 2 * begin : 2 * (begin + valid)]
    expected_down = torch.zeros((2, 64, 32), dtype=torch.uint8)
    expected_down[:, :, : valid // 2] = tensors["down.blocks"].reshape(2, 64, 48)[
        :, :, begin // 2 : (begin + valid) // 2
    ]
    expected_gate_scales = torch.full((2, 128, 2), 127, dtype=torch.uint8)
    expected_gate_scales[:, : 2 * valid] = tensors["gate.scales"][:, 2 * begin : 2 * (begin + valid)]
    expected_down_scales = torch.full((2, 64, 2), 127, dtype=torch.uint8)
    expected_down_scales[:, :, : valid // 32] = tensors["down.scales"][:, :, begin // 32 : (begin + valid) // 32]
    expected_bias = torch.zeros((2, 128), dtype=torch.bfloat16)
    expected_bias[:, : 2 * valid] = tensors["gate.bias"][:, 2 * begin : 2 * (begin + valid)]
    for actual_tensor, expected in zip(
        actual.resources(),
        (
            expected_gate,
            expected_down,
            expected_gate_scales,
            expected_down_scales,
            expected_bias,
            tensors["down.bias"] if rank == 0 else torch.zeros_like(tensors["down.bias"]),
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual_tensor.cpu(), expected, rtol=0, atol=0)
    if rank == 0:
        assert actual.router is not None and actual.router.projection_bias is not None
        torch.testing.assert_close(actual.router.weight.cpu(), tensors["router.weight"], rtol=0, atol=0)
        torch.testing.assert_close(actual.router.projection_bias.cpu(), tensors["router.bias"], rtol=0, atol=0)
    else:
        assert actual.router is None


@pytest.mark.parametrize("failure", ("packed-dtype", "bias-dtype", "packed-layout", "nan-scale"))
@xtest.requirements(device_count=1)
def test_official_layout_validation_fails_closed(tmp_path: Path, failure: str) -> None:
    install_test_config(minimal_config())
    tensors = checkpoint_tensors()
    if failure == "packed-dtype":
        tensors["gate.blocks"] = tensors["gate.blocks"].to(torch.int8)
    elif failure == "bias-dtype":
        tensors["down.bias"] = tensors["down.bias"].float()
    elif failure == "packed-layout":
        tensors["gate.blocks"] = tensors["gate.blocks"].reshape(2, 192, 32)
    else:
        tensors["down.scales"][0, 0, 0] = 255
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises((RuntimeError, ValueError), match="expected|NaN"):
        loader.materialize_local_layer_weights(requests=(request(tmp_path, 0),))
