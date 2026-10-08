"""Fresh-process Router-only diagnostic under the production cuBLAS workspace policy."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

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


def run_production_router_probe(
    *,
    workdir: Path,
    tensors: dict[str, torch.Tensor],
    layer_ids: tuple[int, ...],
    row_counts: tuple[int, ...],
    timeout_seconds: float,
) -> dict[str, torch.Tensor]:
    """Own a fresh child so cuBLAS observes the FfnAgent policy before its first CUDA initialization."""

    deadline = time.monotonic() + timeout_seconds
    workdir.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(tensors, workdir / "inputs.safetensors")
    job = RouterProjectionJob(workdir, layer_ids, row_counts)
    child = PythonChildProcess(
        "router-production-probe", run_router_projection_child, job, log_path=workdir / "child.log"
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


def run_router_projection_child(connection: Connection, job: RouterProjectionJob) -> None:
    """Evaluate exact and padded shapes without constructing a model or changing production code."""

    if torch.cuda.is_initialized():
        raise RuntimeError("Router probe CUDA initialized before the FfnAgent cuBLAS workspace policy")
    # Keep this diagnostic policy aligned with FfnAgent.__init__; changing it
    # in an already initialized parent cannot test cuBLAS workspace selection.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    from xpool.runtime.ffnagent import execution, weights
    from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter

    progress = diagnostics.ReferenceDiagnostics(job.workdir, 0, True)
    outputs: dict[str, torch.Tensor] = {}
    samples: dict[str, object] = {}
    with progress.watchdog(), progress.phase("projection_probe"), torch.inference_mode():
        torch.cuda.set_device(0)
        (job.workdir / "environment.json").write_text(
            json.dumps(diagnostics.cuda_gemm_environment(), indent=2) + "\n", encoding="utf-8"
        )
        tensors = safetensors.torch.load_file(job.workdir / "inputs.safetensors")
        for layer_id in job.layer_ids:
            router = weights.MoeRouterWeights(
                weight=tensors[f"layer-{layer_id}.weight"].cuda(),
                correction_bias=None,
                projection_bias=tensors[f"layer-{layer_id}.bias"].cuda(),
            )
            for rows in job.row_counts:
                case_id = f"layer-{layer_id}-rows-{rows}"
                with progress.phase("projection_case", case_id=case_id):
                    capacity = next(
                        value for value in execution.derive_payload_row_capacities(job.row_counts[-1]) if value >= rows
                    )
                    for name, size in (("production_direct", rows), ("capacity_graph", capacity)):
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
                        if name == "capacity_graph":
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

                        diagnostics.profile_projection(
                            job.workdir / f"{case_id}.{name}.trace.json", {name: project}
                        )
                        outputs[f"{case_id}.{name}_logits"] = logits[:rows].cpu()
                        outputs[f"{case_id}.{name}_ids"] = ids[:rows].cpu()
                        outputs[f"{case_id}.{name}_weights"] = route_weights[:rows].cpu()
                        samples[f"{case_id}.{name}"] = {
                            "row_count": rows,
                            "capacity": size,
                            "input": diagnostics.tensor_geometry(hidden),
                            "weight": diagnostics.tensor_geometry(router.weight),
                            "workspace": diagnostics.tensor_geometry(workspace),
                            "logits": diagnostics.tensor_geometry(logits),
                        }
                    assert router.projection_bias is not None
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
