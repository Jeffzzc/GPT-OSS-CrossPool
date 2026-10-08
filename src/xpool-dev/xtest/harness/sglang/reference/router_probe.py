"""Fresh-process Router diagnostics isolating cuBLAS workspace policy and Graph shape."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

import safetensors.torch
import torch

from xkit.child import PythonChildProcess
from xtest.harness.sglang.reference import diagnostics, ffn_protocol
from xtest.harness.support.wait import remaining_seconds


@dataclass(frozen=True, slots=True)
class RouterProjectionJob:
    """CPU checkpoint Router tensors and shared input prefixes for a bounded diagnostic child."""

    workdir: Path
    layer_ids: tuple[int, ...]
    row_counts: tuple[int, ...]
    workspace_policy: Literal["unset", ":0:0"]


def run_router_projection_probe(
    *,
    workdir: Path,
    tensors: dict[str, torch.Tensor],
    layer_ids: tuple[int, ...],
    row_counts: tuple[int, ...],
    timeout_seconds: float,
    workspace_policy: Literal["unset", ":0:0"],
) -> dict[str, torch.Tensor]:
    """Own one fresh child with an explicit policy installed before its first CUDA initialization."""

    deadline = time.monotonic() + timeout_seconds
    workdir.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(tensors, workdir / "inputs.safetensors")
    job = RouterProjectionJob(workdir, layer_ids, row_counts, workspace_policy)
    child = PythonChildProcess(
        "router-projection-probe", run_router_projection_child, job, log_path=workdir / "child.log"
    )
    try:
        child.start()
        completed = child.receive(
            ffn_protocol.FfnReferenceCompleted, timeout_seconds=remaining_seconds(deadline, "Router probe")
        )
        if completed.case_count != len(layer_ids) * len(row_counts):
            raise RuntimeError("Router probe returned an incomplete case batch")
        child.wait(timeout_seconds=remaining_seconds(deadline, "Router probe exit"))
        return safetensors.torch.load_file(workdir / "outputs.safetensors")
    finally:
        if child.process.is_alive():
            PythonChildProcess.terminate_all((child,))
        child.close()


def install_router_workspace_policy(policy: Literal["unset", ":0:0"]) -> None:
    """Install one diagnostic policy in a fresh child; reject initialized CUDA before mutating its environment."""

    if torch.cuda.is_initialized():
        raise RuntimeError("Router probe CUDA initialized before installing its cuBLAS workspace policy")
    if policy == "unset":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = policy


def projection_variants(rows: int, capacity: int) -> tuple[tuple[str, int, bool], ...]:
    """Return eager/Graph controls at exact and fixed shapes; the caller supplies the derived capacity."""

    return (
        ("production_direct", rows, False),
        ("exact_graph", rows, True),
        ("capacity_graph", capacity, True),
        *(
            (f"capacity_{size}_{mode}", size, mode == "graph")
            for size in (32, 64)
            if size >= rows
            for mode in ("eager", "graph")
        ),
    )


def compare_probe_samples(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    expected_sample: str,
    actual_sample: str,
    rows: int,
) -> dict[str, object]:
    """Compare raw probe prefixes and associate weights by Expert ID; this adds no acceptance threshold."""

    return {
        "logits": diagnostics.tensor_difference(
            expected[f"{expected_sample}_logits"][:rows], actual[f"{actual_sample}_logits"][:rows]
        ),
        "routing": diagnostics.routing_difference(
            expected[f"{expected_sample}_ids"][:rows],
            expected[f"{expected_sample}_weights"][:rows],
            actual[f"{actual_sample}_ids"][:rows],
            actual[f"{actual_sample}_weights"][:rows],
        ),
    }


def summarize_workspace_ab(
    *,
    reference: dict[str, torch.Tensor],
    unset: dict[str, torch.Tensor],
    zero: dict[str, torch.Tensor],
    layer_ids: tuple[int, ...],
    row_counts: tuple[int, ...],
) -> dict[str, object]:
    """Separate workspace changes at identical shapes from shape and capture changes within each policy."""

    samples = {}
    for layer_id in layer_ids:
        for rows in row_counts:
            case_id = f"layer-{layer_id}-rows-{rows}"
            variants = sorted(
                key.removeprefix(f"{case_id}.").removesuffix("_ids")
                for key in unset
                if key.startswith(f"{case_id}.") and key.endswith("_ids")
            )
            workspace = {
                name: compare_probe_samples(
                    unset, zero, expected_sample=f"{case_id}.{name}", actual_sample=f"{case_id}.{name}", rows=rows
                )
                for name in variants
            }
            for name in ("production_linear", "production_addmm_out"):
                workspace[name] = {
                    "logits": diagnostics.tensor_difference(
                        unset[f"{case_id}.{name}_logits"], zero[f"{case_id}.{name}_logits"]
                    )
                }
            shape = {}
            for policy, values in (("unset", unset), (":0:0", zero)):
                controls = {
                    f"batch_{row_counts[-1]}_prefix": (
                        f"layer-{layer_id}-rows-{row_counts[-1]}.production_direct",
                        "production_direct",
                    ),
                    "eager_vs_exact_graph": (f"{case_id}.production_direct", "exact_graph"),
                    "exact_vs_selected_capacity": (f"{case_id}.production_direct", "capacity_graph"),
                }
                for capacity in (32, 64):
                    if capacity >= rows:
                        controls[f"exact_vs_capacity_{capacity}_eager"] = (
                            f"{case_id}.production_direct",
                            f"capacity_{capacity}_eager",
                        )
                        controls[f"capacity_{capacity}_eager_vs_graph"] = (
                            f"{case_id}.capacity_{capacity}_eager",
                            f"capacity_{capacity}_graph",
                        )
                shape[policy] = {
                    name: compare_probe_samples(
                        values,
                        values,
                        expected_sample=expected_sample,
                        actual_sample=f"{case_id}.{actual_name}",
                        rows=rows,
                    )
                    for name, (expected_sample, actual_name) in controls.items()
                }
            reference_by_policy = {
                policy: {
                    name: compare_probe_samples(
                        reference,
                        values,
                        expected_sample=f"{case_id}.reference",
                        actual_sample=f"{case_id}.{name}",
                        rows=rows,
                    )
                    for name in variants
                }
                for policy, values in (("unset", unset), (":0:0", zero))
            }
            samples[case_id] = {
                "workspace_unset_vs_zero": workspace,
                "shape_by_policy": shape,
                "reference_vs_policy": reference_by_policy,
            }
    return {"samples": samples}


def run_router_projection_child(connection: Connection, job: RouterProjectionJob) -> None:
    """Evaluate exact and padded shapes without constructing a model or changing production code."""

    install_router_workspace_policy(job.workspace_policy)
    from xpool.native import ffnagent
    from xpool.runtime.ffnagent import execution, weights
    from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter

    progress = diagnostics.ReferenceDiagnostics(job.workdir, 0, True)
    outputs: dict[str, torch.Tensor] = {}
    samples: dict[str, object] = {}
    with (
        progress.watchdog(),
        progress.phase("projection_probe", workspace_policy=job.workspace_policy),
        torch.inference_mode(),
    ):
        torch.cuda.set_device(0)
        (job.workdir / "environment.json").write_text(
            json.dumps(diagnostics.cuda_gemm_environment(include_workspace_limits=True), indent=2) + "\n",
            encoding="utf-8",
        )
        tensors = safetensors.torch.load_file(job.workdir / "inputs.safetensors")
        for layer_id in job.layer_ids:
            router = weights.MoeRouterWeights(
                weight=tensors[f"layer-{layer_id}.weight"].cuda(),
                correction_bias=None,
                projection_bias=tensors[f"layer-{layer_id}.bias"].cuda(),
            )
            assert router.projection_bias is not None
            for rows in job.row_counts:
                case_id = f"layer-{layer_id}-rows-{rows}"
                with progress.phase("projection_case", case_id=case_id):
                    capacity = next(
                        value for value in execution.derive_payload_row_capacities(job.row_counts[-1]) if value >= rows
                    )
                    for name, size, captured in projection_variants(rows, capacity):
                        hidden = torch.zeros((size, 2880), device="cuda", dtype=torch.bfloat16)
                        hidden[:rows].copy_(tensors["hidden_states"][:rows])
                        workspace = torch.empty(
                            GptOssAdapter.router_workspace_bytes(
                                payload_dtype=torch.bfloat16,
                                payload_row_capacity=size,
                                hidden_size=2880,
                                routed_expert_count=32,
                                routed_topk=4,
                            ),
                            device="cuda",
                            dtype=torch.uint8,
                        )
                        ids = torch.empty((size, 4), device="cuda", dtype=torch.int32)
                        route_weights = torch.empty((size, 4), device="cuda", dtype=torch.float32)

                        def compute() -> None:
                            GptOssAdapter.compute_routed_topk(
                                hidden_states=hidden,
                                router_weights=router,
                                workspace=workspace,
                                routed_ids=ids,
                                routed_weights=route_weights,
                                renormalize=True,
                            )

                        logits = workspace[: size * 32 * 2].view(torch.bfloat16).view(size, 32)
                        if captured:
                            graph = torch.cuda.CUDAGraph()
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                compute()
                            torch.cuda.current_stream().wait_stream(stream)
                            with torch.cuda.graph(graph, stream=stream):
                                compute()

                            def project() -> torch.Tensor:
                                graph.replay()
                                return logits

                        else:

                            def project() -> torch.Tensor:
                                compute()
                                return logits

                        trace_path = job.workdir / f"{case_id}.{name}.trace.json"
                        diagnostics.profile_projection(trace_path, {name: project})
                        outputs[f"{case_id}.{name}_logits"] = logits[:rows].cpu()
                        outputs[f"{case_id}.{name}_ids"] = ids[:rows].cpu()
                        outputs[f"{case_id}.{name}_weights"] = route_weights[:rows].cpu()
                        samples[f"{case_id}.{name}"] = {
                            "row_count": rows,
                            "capacity": size,
                            "captured": captured,
                            "workspace_policy": job.workspace_policy,
                            "input": diagnostics.tensor_geometry(hidden),
                            "weight": diagnostics.tensor_geometry(router.weight),
                            "bias": diagnostics.tensor_geometry(router.projection_bias),
                            "workspace": diagnostics.tensor_geometry(workspace),
                            "explicit_lt_scratch": diagnostics.tensor_geometry(
                                workspace[GptOssAdapter.router_workspace_layout(payload_row_capacity=size)[2] :]
                            ),
                            "explicit_lt_budget_bytes": ffnagent.BIASED_ROUTER_GEMM_WORKSPACE_BYTES,
                            "logits": diagnostics.tensor_geometry(logits),
                            "kernels": diagnostics.cuda_kernel_inventory(trace_path),
                        }
                    # These calls run only after graph replay, outside capture.
                    exact_hidden = tensors["hidden_states"][:rows].cuda()
                    destination = torch.empty((rows, 32), device="cuda", dtype=torch.bfloat16)
                    candidates = diagnostics.profile_projection(
                        job.workdir / f"{case_id}.trace.json",
                        {
                            "linear": lambda: torch.nn.functional.linear(
                                exact_hidden, router.weight, router.projection_bias
                            ),
                            "addmm_out": lambda: torch.addmm(
                                router.projection_bias, exact_hidden, router.weight.t(), out=destination
                            ),
                        },
                    )
                    for candidate, value in candidates.items():
                        outputs[f"{case_id}.production_{candidate}_logits"] = value.cpu()
                    staging_path = job.workdir / "outputs.safetensors.partial"
                    safetensors.torch.save_file(outputs, staging_path)
                    staging_path.replace(job.workdir / "outputs.safetensors")
                    (job.workdir / "geometry.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    connection.send(ffn_protocol.FfnReferenceCompleted(len(job.layer_ids) * len(job.row_counts)))
