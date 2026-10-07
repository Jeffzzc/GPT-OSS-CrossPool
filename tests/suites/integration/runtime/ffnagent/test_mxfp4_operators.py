"""Independent packed decode, biased epilogues and allocation-free Graph replay."""

from __future__ import annotations

import pytest
import torch

import xtest
from xpool import ffn
from xpool.runtime.ffnagent import execution, operators, registry, weights


def decode_reference(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 and UE8M0 independently with PyTorch, never a production operator."""

    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=blocks.device, dtype=torch.float32
    )
    nibbles = torch.stack((blocks & 15, blocks >> 4), dim=-1).long().flatten(-2)
    decoded = table[nibbles] * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return decoded.to(torch.bfloat16)


def independent_reference(
    hidden_states: torch.Tensor,
    layer: weights.Mxfp4MoeFfnWeights,
    ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain FP32 W13/bias through activation and BF16 weighted W2 before combine."""

    w13 = decode_reference(layer.gate_up_blocks, layer.gate_up_scales).float()
    w2 = decode_reference(layer.down_blocks, layer.down_scales).float()
    gate_up = torch.zeros((ids.numel(), w13.shape[1]), device="cuda", dtype=torch.float32)
    activated = torch.zeros((ids.numel(), w13.shape[1] // 2), device="cuda", dtype=torch.bfloat16)
    outputs = torch.zeros((*ids.shape, hidden_states.shape[1]), device="cuda", dtype=torch.bfloat16)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            if expert < 0:
                continue
            index = row * ids.shape[1] + slot
            values = hidden_states[row].float() @ w13[expert].t() + layer.gate_up_bias[expert].float()
            gate_up[index] = values
            gate = values[::2].clamp(max=7)
            up = values[1::2].clamp(-7, 7)
            activated[index] = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            outputs[row, slot] = (
                activated[index].float() @ w2[expert].t() + layer.down_bias[expert].float()
            ) * route_weights[row, slot]
    return gate_up, activated, outputs, outputs.float().sum(dim=1).to(torch.bfloat16)


@pytest.mark.parametrize("rows", (1, 33))
@xtest.requirements(device_count=1)
def test_mxfp4_matmuls_activation_bias_and_replay(rows: int) -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16) * 0.25
    layer = weights.Mxfp4MoeFfnWeights(
        gate_up_blocks=torch.randint(0, 256, (2, 128, 32), device="cuda", dtype=torch.uint8),
        down_blocks=torch.randint(0, 256, (2, 64, 32), device="cuda", dtype=torch.uint8),
        gate_up_scales=torch.randint(122, 125, (2, 128, 2), device="cuda", dtype=torch.uint8),
        down_scales=torch.randint(122, 125, (2, 64, 2), device="cuda", dtype=torch.uint8),
        gate_up_bias=torch.randn((2, 128), device="cuda", dtype=torch.bfloat16),
        down_bias=torch.randn((2, 64), device="cuda", dtype=torch.bfloat16),
        router=None,
    )
    ids = torch.tensor([[0, 1]] * rows, device="cuda", dtype=torch.int32)
    route_weights = torch.tensor([[0.25, 0.75]] * rows, device="cuda", dtype=torch.float32)
    if rows > 1:
        ids[-1] = -1
        route_weights[-1] = 0
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=rows,
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
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    workspace = registry.allocate_moe_graph_capture_workspace(signature=signature, w13_config=config)
    output = torch.empty_like(hidden_states)

    def compute() -> torch.Tensor:
        return operators.compute_moe_partial(
            hidden_states=hidden_states,
            layer_weights=layer,
            topk_ids=ids,
            topk_weights=route_weights,
            sorted_token_ids=workspace.sorted_token_ids,
            expert_ids=workspace.expert_ids,
            num_tokens_post_padded=workspace.num_tokens_post_padded,
            cumsum_buffer=workspace.cumsum_buffer,
            gate_up=workspace.gate_up,
            activated=workspace.activated,
            route_outputs=workspace.route_outputs,
            output=output,
            w13_config=config,
            w2_config=None,
            activation=signature.activation,
            activation_alpha=signature.activation_alpha,
            activation_clamp_limit=signature.activation_clamp_limit,
            routed_scaling_factor=1.0,
        )

    expected_gate_up, expected_activation, expected_routes, expected = independent_reference(
        hidden_states, layer, ids, route_weights
    )
    assert compute() is output
    torch.testing.assert_close(workspace.route_outputs, expected_routes, rtol=0.008, atol=0.008)
    # W13/W2 share storage; test W13 directly to locate decode/layout/bias errors.
    operators.mxfp4_expert_matmul_kernel[(workspace.expert_ids.numel(), 2)](
        hidden_states,
        layer.gate_up_blocks,
        layer.gate_up_scales,
        layer.gate_up_bias,
        workspace.gate_up,
        route_weights,
        workspace.sorted_token_ids,
        workspace.expert_ids,
        workspace.num_tokens_post_padded,
        N=128,
        K=64,
        EXPERT_COUNT=2,
        ROUTE_COUNT=rows * 2,
        INPUT_TOPK=2,
        MULTIPLY_ROUTE=False,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    torch.testing.assert_close(workspace.gate_up, expected_gate_up, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(workspace.activated, expected_activation, rtol=0.008, atol=0.008)
    torch.testing.assert_close(output, expected, rtol=0.008, atol=0.008)
    compute()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compute()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute()
    hidden_states.mul_(0.5)
    layer.gate_up_bias.fill_(8)
    layer.down_bias.fill_(-0.5)
    layer.down_scales.fill_(123)
    graph.replay()
    _, expected_activation, expected_routes, expected = independent_reference(hidden_states, layer, ids, route_weights)
    torch.testing.assert_close(workspace.route_outputs, expected_routes, rtol=0.008, atol=0.008)
    torch.testing.assert_close(workspace.activated, expected_activation, rtol=0.008, atol=0.008)
    torch.testing.assert_close(output, expected, rtol=0.008, atol=0.008)
    if rows > 1:
        assert torch.count_nonzero(output[-1]) == 0


@xtest.requirements(device_count=1)
def test_clamped_activation_matches_pinned_fused_epilogue() -> None:
    """Use the original pinned activation, including asymmetric clamp and fused multiply-add."""

    import triton
    from triton import language
    from triton_kernels.swiglu import swiglu_fn

    @triton.jit
    def reference_kernel(input_pointer, output_pointer):
        offsets = language.arange(0, 256)
        values = language.load(input_pointer + offsets).reshape(1, 256)
        output = swiglu_fn(values, 1.702, 7.0).reshape(128)
        language.store(output_pointer + language.arange(0, 128), output)

    values = torch.linspace(-12, 12, 256, device="cuda", dtype=torch.float32).view(1, 256)
    output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    reference_kernel[(1,)](values, expected, enable_fp_fusion=False)
    operators.compute_clamped_swiglu(gate_up=values, output=output, alpha=1.702, clamp_limit=7)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
