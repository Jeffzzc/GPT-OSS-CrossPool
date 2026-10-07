"""Biased BF16 Router cutoff/tie behavior, capture and caller-owned storage."""

from __future__ import annotations

import pytest
import torch
from triton_kernels.topk import topk_forward

import xtest
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter


@pytest.mark.parametrize("rows", (1, 31, 32, 33, 129))
@xtest.requirements(device_count=1)
def test_cutoff_ties_and_graph_capture(rows: int) -> None:
    hidden_states = torch.zeros((rows, 2880), device="cuda", dtype=torch.bfloat16)
    router = weights.MoeRouterWeights(
        weight=torch.zeros((32, 2880), device="cuda", dtype=torch.bfloat16),
        correction_bias=None,
        projection_bias=torch.tensor(
            [3.0, 2.0, 1.0, 1.0, 1.0, 0.99609375] + [-2.0] * 26, device="cuda", dtype=torch.bfloat16
        ),
    )
    workspace = torch.empty(
        GptOssAdapter.router_workspace_bytes(
            payload_dtype=torch.bfloat16,
            payload_row_capacity=rows,
            hidden_size=2880,
            routed_expert_count=32,
            routed_topk=4,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    ids = torch.empty((rows, 4), device="cuda", dtype=torch.int32)
    route_weights = torch.empty((rows, 4), device="cuda", dtype=torch.float32)

    def compute() -> None:
        GptOssAdapter.compute_routed_topk(
            hidden_states=hidden_states,
            router_weights=router,
            workspace=workspace,
            routed_ids=ids,
            routed_weights=route_weights,
            renormalize=True,
        )

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
    assert router.projection_bias is not None
    router.projection_bias[5] = 4.0
    graph.replay()
    expected_logits = torch.nn.functional.linear(hidden_states, router.weight, router.projection_bias)
    expected_weights, expected_ids, _ = topk_forward(expected_logits, 4)
    actual_logits = workspace[: rows * 32 * 2].view(torch.bfloat16).view(rows, 32)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(route_weights, expected_weights.float(), rtol=0, atol=0)
    assert ids[0].tolist() == [5, 0, 1, 2]
