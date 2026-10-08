"""Opt-in reference progress logs and Router GEMM evidence outside captured execution."""

from __future__ import annotations

import contextlib
import faulthandler
import importlib.metadata
import inspect
import json
import os
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class ReferenceDiagnostics:
    """Flush rank-local progress before blocking operations, preserving a cancelled run's prefix."""

    workdir: Path
    rank: int | str
    enabled: bool

    def event(self, phase: str, event: str, **details: object) -> None:
        """Append JSON-serializable diagnostic fields; never inspect arbitrary environment secrets."""

        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": self.rank,
            "phase": phase,
            "event": event,
            **details,
        }
        with (self.workdir / f"rank-{self.rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()

    @contextlib.contextmanager
    def phase(self, name: str, **details: object) -> Generator[None]:
        """Record entry, completion or failure without adding CUDA synchronization."""

        self.event(name, "start", **details)
        try:
            yield
        except BaseException as error:
            self.event(name, "error", error=repr(error), **details)
            raise
        else:
            self.event(name, "complete", **details)

    @contextlib.contextmanager
    def watchdog(self) -> Generator[None]:
        """Dump Python stacks every 120 seconds until the bounded reference job exits."""

        if not self.enabled:
            yield
            return
        with (self.workdir / f"rank-{self.rank}-stacks.log").open("a", encoding="utf-8") as stream:
            faulthandler.dump_traceback_later(120, repeat=True, file=stream)
            try:
                yield
            finally:
                faulthandler.cancel_dump_traceback_later()


def tensor_geometry(tensor: torch.Tensor) -> dict[str, object]:
    """Describe actual GEMM shape, layout and pointer alignment without reading tensor values."""

    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "alignment_mod_256": tensor.data_ptr() % 256,
        "requires_grad": tensor.requires_grad,
    }


def cuda_gemm_environment() -> dict[str, object]:
    """Record relevant GEMM controls without changing flags, workspaces or backend selection."""

    device = torch.cuda.current_device()
    versions: dict[str, str | None] = {}
    for name in ("sglang", "torch", "triton", "nvidia-cublas-cu13", "nvidia-cublas-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "pid": os.getpid(),
        "device_index": device,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUBLAS_WORKSPACE_SIZE",
                "CUBLASLT_WORKSPACE_SIZE",
                "DISABLE_ADDMM_CUDA_LT",
                "NVIDIA_TF32_OVERRIDE",
                "SGLANG_ENABLE_BF16_SPLITK_GEMM",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
                "TORCH_DISTRIBUTED_DEBUG",
            )
        },
    }


def tensor_difference(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    """Count exact CPU element mismatches and retain at most eight coordinates and values."""

    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.shape != actual.shape:
        raise ValueError("Router diagnostic tensors must have identical shapes")
    coordinates = (expected != actual).nonzero()
    return {
        "shape": list(expected.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "different_elements": coordinates.shape[0],
        "max_abs_error": float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0,
        "first_mismatches": [
            {
                "index": coordinate.tolist(),
                "expected": float(expected[tuple(coordinate.tolist())]),
                "actual": float(actual[tuple(coordinate.tolist())]),
            }
            for coordinate in coordinates[:8]
        ],
    }


def routing_difference(
    expected_ids: torch.Tensor,
    expected_weights: torch.Tensor,
    actual_ids: torch.Tensor,
    actual_weights: torch.Tensor,
) -> dict[str, object]:
    """Compare weights by Expert ID while retaining both ordered and set mismatch counts."""

    expected_sorted, expected_order = expected_ids.sort(dim=-1)
    actual_sorted, actual_order = actual_ids.sort(dim=-1)
    matching_rows = (expected_sorted == actual_sorted).all(dim=-1)
    return {
        "ordered_ids": tensor_difference(expected_ids, actual_ids),
        "sorted_ids": tensor_difference(expected_sorted, actual_sorted),
        "matched_expert_rows": int(matching_rows.sum()),
        "weights_by_expert_id": tensor_difference(
            expected_weights.gather(-1, expected_order)[matching_rows],
            actual_weights.gather(-1, actual_order)[matching_rows],
        ),
    }


def profile_projection(path: Path, calls: dict[str, Callable[[], torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Profile diagnostic-only eager calls; never enter this helper from a production graph."""

    outputs = {}
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        for name, call in calls.items():
            with torch.profiler.record_function(name):
                outputs[name] = call()
    profile.export_chrome_trace(str(path))
    return outputs


def module_implementation(module: torch.nn.Module) -> dict[str, object]:
    """Retain the installed Router and quant-method sources rather than infer dispatch from a model name."""

    methods = {"forward": type(module).forward}
    for parent in type(module).__mro__[1:]:
        if parent.__module__.startswith("sglang.") and "forward" in parent.__dict__:
            methods[f"{parent.__name__}.forward"] = parent.forward
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        methods["quant_method.apply"] = type(quant_method).apply
    return {
        name: {"qualified_name": f"{method.__module__}.{method.__qualname__}", "source": inspect.getsource(method)}
        for name, method in methods.items()
    }
