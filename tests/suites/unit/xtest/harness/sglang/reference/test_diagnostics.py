from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xtest.harness.sglang.reference.diagnostics import ReferenceDiagnostics, routing_difference, tensor_difference


def test_disabled_diagnostics_create_no_artifacts(tmp_path: Path) -> None:
    progress = ReferenceDiagnostics(tmp_path / "absent", 0, False)
    with progress.phase("model_loader"), progress.watchdog():
        progress.event("cuda_device_binding", "complete")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("error", (RuntimeError("load failed"), KeyboardInterrupt("cancelled")))
def test_rank_progress_survives_failure_and_cancellation(tmp_path: Path, error: BaseException) -> None:
    progress = ReferenceDiagnostics(tmp_path, 1, True)
    with pytest.raises(type(error), match=str(error)), progress.phase("model_loader", layer_id=12):
        assert (tmp_path / "rank-1.jsonl").read_text(encoding="utf-8")
        raise error
    records = [json.loads(line) for line in (tmp_path / "rank-1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["start", "error"]
    assert all(record["rank"] == 1 and record["pid"] > 0 and record["layer_id"] == 12 for record in records)
    assert records[0]["monotonic_ns"] <= records[1]["monotonic_ns"]
    assert all(record["timestamp"].endswith("+00:00") for record in records)
    assert str(error) in records[1]["error"]


def test_tensor_difference_retains_exact_bf16_mismatches() -> None:
    expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1, 0] = 3.125
    result = tensor_difference(expected, actual)
    assert result["different_elements"] == 1
    assert result["max_abs_error"] == 0.125
    assert result["first_mismatches"] == [{"index": [1, 0], "expected": 3.0, "actual": 3.125}]


def test_routing_difference_associates_weights_by_expert_id() -> None:
    expected_ids = torch.tensor([[3, 1], [0, 2]])
    actual_ids = torch.tensor([[1, 3], [1, 2]])
    expected_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    actual_weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    result = routing_difference(expected_ids, expected_weights, actual_ids, actual_weights)
    assert result["matched_expert_rows"] == 1
    assert result["weights_by_expert_id"] == tensor_difference(
        torch.tensor([[0.75, 0.25]]), torch.tensor([[0.75, 0.25]])
    )
    assert result["sorted_ids"] == tensor_difference(torch.tensor([[1, 3], [0, 2]]), torch.tensor([[1, 3], [1, 2]]))
