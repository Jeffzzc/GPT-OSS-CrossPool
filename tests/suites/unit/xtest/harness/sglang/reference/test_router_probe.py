from __future__ import annotations

import json
import os
from typing import Literal

import pytest
import torch

from xtest.harness.sglang.reference import router_probe


@pytest.mark.parametrize("policy", ("unset", ":0:0"))
def test_workspace_policy_is_explicit_and_preserves_other_gemm_controls(
    monkeypatch: pytest.MonkeyPatch, policy: Literal["unset", ":0:0"]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUBLASLT_WORKSPACE_SIZE", "1024")
    monkeypatch.setenv("TORCH_CUBLASLT_UNIFIED_WORKSPACE", "1")
    router_probe.install_router_workspace_policy(policy)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (None if policy == "unset" else ":0:0")
    assert os.environ["CUBLASLT_WORKSPACE_SIZE"] == "1024"
    assert os.environ["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] == "1"


def test_workspace_policy_rejects_initialized_cuda_before_environment_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="CUDA initialized"):
        router_probe.install_router_workspace_policy("unset")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("rows,capacity", ((1, 1), (31, 32), (32, 32), (33, 64), (4096, 4096)))
def test_projection_matrix_separates_shape_from_capture(rows: int, capacity: int) -> None:
    variants = router_probe.projection_variants(rows, capacity)
    assert ("production_direct", rows, False) in variants
    assert ("exact_graph", rows, True) in variants
    assert ("capacity_graph", capacity, True) in variants
    assert all(size >= rows for _, size, _ in variants)
    for fixed_capacity in (32, 64):
        if fixed_capacity >= rows:
            assert (f"capacity_{fixed_capacity}_eager", fixed_capacity, False) in variants
            assert (f"capacity_{fixed_capacity}_graph", fixed_capacity, True) in variants


def test_workspace_summary_separates_environment_and_shape_effects() -> None:
    unset: dict[str, torch.Tensor] = {}
    for rows in (1, 2):
        for name, _, _ in router_probe.projection_variants(rows, rows):
            prefix = f"layer-0-rows-{rows}.{name}"
            unset[f"{prefix}_logits"] = torch.tensor([[1.0, 2.0]]).repeat(rows, 1).bfloat16()
            unset[f"{prefix}_ids"] = torch.tensor([[0, 1]]).repeat(rows, 1)
            unset[f"{prefix}_weights"] = torch.tensor([[0.25, 0.75]]).repeat(rows, 1)
        for name in ("production_linear", "production_addmm_out"):
            unset[f"layer-0-rows-{rows}.{name}_logits"] = unset[f"layer-0-rows-{rows}.production_direct_logits"].clone()
    zero = {key: value.clone() for key, value in unset.items()}
    zero["layer-0-rows-1.production_direct_logits"][0, 0] += 0.125
    zero["layer-0-rows-1.production_direct_ids"] = torch.tensor([[1, 0]])
    zero["layer-0-rows-1.production_direct_weights"] = torch.tensor([[0.75, 0.25]])
    reference = {
        f"layer-0-rows-{rows}.reference_{kind}": unset[f"layer-0-rows-{rows}.production_direct_{kind}"]
        for rows in (1, 2)
        for kind in ("logits", "ids", "weights")
    }
    # Inspect the actual JSON surface consumed by artifact analysis.
    samples = json.loads(
        json.dumps(
            router_probe.summarize_workspace_ab(
                reference=reference, unset=unset, zero=zero, layer_ids=(0,), row_counts=(1, 2)
            )
        )
    )["samples"]
    workspace = samples["layer-0-rows-1"]["workspace_unset_vs_zero"]
    assert workspace["production_direct"]["logits"]["max_abs_error"] == 0.125
    assert workspace["production_direct"]["routing"]["weights_by_expert_id"]["different_elements"] == 0
    assert workspace["capacity_32_eager"]["logits"]["different_elements"] == 0
    shape = samples["layer-0-rows-1"]["shape_by_policy"]
    assert shape["unset"]["exact_vs_capacity_32_eager"]["logits"]["different_elements"] == 0
    assert shape[":0:0"]["exact_vs_capacity_32_eager"]["logits"]["max_abs_error"] == 0.125
    reference_by_policy = samples["layer-0-rows-1"]["reference_vs_policy"]
    assert reference_by_policy["unset"]["production_direct"]["logits"]["different_elements"] == 0
    assert reference_by_policy[":0:0"]["production_direct"]["logits"]["max_abs_error"] == 0.125
