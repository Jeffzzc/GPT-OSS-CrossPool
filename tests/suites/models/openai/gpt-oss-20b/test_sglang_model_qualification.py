"""Official MXFP4 GPT-OSS-20B numerical, routing and serving qualification."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import pytest
import safetensors.torch
import torch
from safetensors import safe_open
from tests import TEST_CATALOG_PATH

import xtest
from xkit.deployment import resolve_deployment_path
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId
from xpool.runtime.ffnagent import architecture, checkpoint, weights
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase, E2eServingCase
from xtest.harness.sglang.reference import diagnostics, router_probe
from xtest.harness.sglang.reference.ffn import SglangFfnReferenceCase, SglangFfnReferenceRunner
from xtest.harness.sglang.serving import qualification
from xtest.harness.support.ffn import generate_ffn_hidden_states
from xtest.harness.support.wait import remaining_seconds

pytest_plugins = ("xtest.harness.support.config",)

MODEL = ModelId("openai/gpt-oss-20b")
LAYERS = (0, 12, 23)
ROWS = (1, 31, 32, 33, 4096)
ROUTER_TIMEOUT_SECONDS = 1800
ROUTER_HARNESS_RESERVE_SECONDS = 120
NUMERICAL_CASES = (
    E2eFfnNumericalCase(
        description="Original pinned SGLang MXFP4 FFN parity across representative layers and capacity boundaries.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        model_id=MODEL,
        layer_ids=LAYERS,
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=ROWS),
        reference_moe_runner_backend="triton_kernel",
        reference_dtype="bfloat16",
        reference_diagnostics=True,
        estimated_duration_seconds=900,
        timeout_seconds=1800,
    ),
)
SERVING_CASES = tuple(
    E2eServingCase(
        description=f"GPT-OSS graph qualification with attention TP 1, DP 1 and FFN TP {tp_size}.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), f"atn1-ffn{tp_size}-lanes1"),
        models=(MODEL,),
        graph_modes=(SglangGraphMode.EAGER, SglangGraphMode.DECODE_FULL, SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
        disable_hybrid_swa_memory=True,
        dtype="bfloat16",
        estimated_duration_seconds=800,
        timeout_seconds=3600,
    )
    for tp_size in (1, 2)
)


@xtest.parameterize("case", tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES))
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare production Graph replay/rebinding with unmodified SGLang FFNs."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)


@xtest.parameterize(("case", "graph_mode"), SERVING_CASES, rows=partial(qualification.graph_rows, compare_modes=True))
@xtest.requirements(qualification.requirements_of)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Require installed graph structure, decode identity and prefill logit/KL parity."""

    qualification.run_serving_case(case, graph_mode, e2e_base_config, tmp_path, task_artifact_dir, compare_modes=True)


@pytest.mark.timeout(ROUTER_TIMEOUT_SECONDS)
@pytest.mark.estimated_duration(seconds=900)
@xtest.requirements(device_count=2, requires_config=True, model_ids=(MODEL,))
def test_checkpoint_router_projection_and_topk(
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Locate routing errors using actual logits and routes from isolated original SGLang."""

    deadline = time.monotonic() + ROUTER_TIMEOUT_SECONDS - ROUTER_HARNESS_RESERVE_SECONDS
    model_path = e2e_base_config.config.model_path_of(MODEL)
    architecture.load(model_id=MODEL, model_path=model_path)
    inputs = generate_ffn_hidden_states(hidden_size=2880, seed=17, row_counts=ROWS)
    cases = tuple(
        SglangFfnReferenceCase(case_id=f"layer-{layer}-rows-{rows}", layer_id=layer, hidden_states=values)
        for layer in LAYERS
        for rows, values in zip(ROWS, inputs, strict=True)
    )
    artifact_dir = task_artifact_dir or tmp_path
    results = SglangFfnReferenceRunner(
        workdir=artifact_dir / "router-reference",
        timeout_seconds=remaining_seconds(deadline, "Router reference"),
    ).run(
        model_path=model_path,
        tensor_parallel_size=2,
        cases=cases,
        moe_runner_backend="triton_kernel",
        dtype="bfloat16",
        diagnostics=True,
    )
    key_view = checkpoint.read_checkpoint_key_view(model_path)
    probe_inputs = {"hidden_states": inputs[-1]}
    for layer_id in LAYERS:
        for suffix in ("weight", "bias"):
            key = f"model.layers.{layer_id}.mlp.router.{suffix}"
            with safe_open(str(key_view[key]), framework="pt", device="cpu") as shard:
                probe_inputs[f"layer-{layer_id}.{suffix}"] = shard.get_tensor(key)
    evidence = router_probe.run_production_router_probe(
        workdir=artifact_dir / "router-production-probe",
        tensors=probe_inputs,
        layer_ids=LAYERS,
        row_counts=ROWS,
        timeout_seconds=remaining_seconds(deadline, "production Router environment probe"),
    )
    samples: dict[str, object] = {}
    (artifact_dir / "gpt-oss-router-environment.json").write_text(
        json.dumps(diagnostics.cuda_gemm_environment(), indent=2) + "\n", encoding="utf-8"
    )
    for case, result in zip(cases, results, strict=True):
        remaining_seconds(deadline, "Router projection probes")
        router = weights.MoeRouterWeights(
            weight=probe_inputs[f"layer-{case.layer_id}.weight"].cuda(),
            correction_bias=None,
            projection_bias=probe_inputs[f"layer-{case.layer_id}.bias"].cuda(),
        )
        hidden_states = case.hidden_states.to(device="cuda")
        row_count = hidden_states.shape[0]
        workspace = torch.empty(
            GptOssAdapter.router_workspace_bytes(
                payload_dtype=torch.bfloat16,
                payload_row_capacity=row_count,
                hidden_size=2880,
                routed_expert_count=32,
                routed_topk=4,
            ),
            device="cuda",
            dtype=torch.uint8,
        )
        ids = torch.empty((row_count, 4), device="cuda", dtype=torch.int32)
        route_weights = torch.empty((row_count, 4), device="cuda", dtype=torch.float32)
        GptOssAdapter.compute_routed_topk(
            hidden_states=hidden_states,
            router_weights=router,
            workspace=workspace,
            routed_ids=ids,
            routed_weights=route_weights,
            renormalize=True,
        )
        logits = workspace[: row_count * 32 * 2].view(torch.bfloat16).view(row_count, 32).cpu()
        assert result.router_logits is not None and result.routing is not None
        evidence[f"{case.case_id}.actual_logits"] = logits
        evidence[f"{case.case_id}.reference_logits"] = result.router_logits
        evidence[f"{case.case_id}.actual_ids"] = ids.cpu()
        evidence[f"{case.case_id}.reference_ids"] = result.routing.topk_ids
        evidence[f"{case.case_id}.actual_weights"] = route_weights.cpu()
        evidence[f"{case.case_id}.reference_weights"] = result.routing.topk_weights
        assert router.projection_bias is not None
        with torch.inference_mode():
            addmm_destination = torch.empty((row_count, 32), device=hidden_states.device, dtype=torch.bfloat16)
            projections = diagnostics.profile_projection(
                artifact_dir / f"{case.case_id}-projection.trace.json",
                {
                    "linear": lambda: torch.nn.functional.linear(hidden_states, router.weight, router.projection_bias),
                    "addmm_out": lambda: torch.addmm(
                        router.projection_bias, hidden_states, router.weight.t(), out=addmm_destination
                    ),
                },
            )
        for name, value in projections.items():
            evidence[f"{case.case_id}.{name}_logits"] = value.cpu()

        samples[case.case_id] = {
            "layer_id": case.layer_id,
            "row_count": row_count,
            "input": diagnostics.tensor_geometry(hidden_states),
            "weight": diagnostics.tensor_geometry(router.weight),
            "bias": diagnostics.tensor_geometry(router.projection_bias),
            "workspace": diagnostics.tensor_geometry(workspace),
            "logits": {
                name: diagnostics.tensor_difference(result.router_logits, evidence[f"{case.case_id}.{name}_logits"])
                for name in (
                    "actual",
                    "linear",
                    "addmm_out",
                    "production_direct",
                    "production_linear",
                    "production_addmm_out",
                    "capacity_graph",
                )
            },
            "routing": diagnostics.routing_difference(
                result.routing.topk_ids, result.routing.topk_weights, ids.cpu(), route_weights.cpu()
            ),
            "capacity_graph_routing": diagnostics.routing_difference(
                result.routing.topk_ids,
                result.routing.topk_weights,
                evidence[f"{case.case_id}.capacity_graph_ids"],
                evidence[f"{case.case_id}.capacity_graph_weights"],
            ),
            "production_direct_routing": diagnostics.routing_difference(
                result.routing.topk_ids,
                result.routing.topk_weights,
                evidence[f"{case.case_id}.production_direct_ids"],
                evidence[f"{case.case_id}.production_direct_weights"],
            ),
            "reference_softmax_reconstruction": diagnostics.tensor_difference(
                result.routing.topk_weights,
                result.router_logits.float().gather(-1, result.routing.topk_ids.long()).softmax(-1).bfloat16().float(),
            ),
            "actual_softmax_reconstruction": diagnostics.tensor_difference(
                route_weights.cpu(), logits.float().gather(-1, ids.cpu().long()).softmax(-1).bfloat16().float(),
            ),
        }
        # Preserve the failing stage's evidence before any parity assertion.
        tensor_staging_path = artifact_dir / "gpt-oss-router.safetensors.partial"
        safetensors.torch.save_file(evidence, tensor_staging_path)
        tensor_staging_path.replace(artifact_dir / "gpt-oss-router.safetensors")
        summary_staging_path = artifact_dir / "gpt-oss-router.json.partial"
        summary_staging_path.write_text(json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8")
        summary_staging_path.replace(artifact_dir / "gpt-oss-router.json")

    prefixes = {}
    for case, result in zip(cases, results, strict=True):
        prefix = f"layer-{case.layer_id}-rows-{ROWS[-1]}"
        row_count = case.hidden_states.shape[0]
        prefixes[case.case_id] = {
            name: diagnostics.tensor_difference(
                evidence[f"{prefix}.{name}_logits"][:row_count], evidence[f"{case.case_id}.{name}_logits"]
            )
            for name in (
                "reference",
                "actual",
                "linear",
                "addmm_out",
                "production_direct",
                "production_linear",
                "production_addmm_out",
                "capacity_graph",
            )
        }
        prefixes[case.case_id]["actual_vs_large_reference_routing"] = diagnostics.routing_difference(
            evidence[f"{prefix}.reference_ids"][:row_count],
            evidence[f"{prefix}.reference_weights"][:row_count],
            evidence[f"{case.case_id}.actual_ids"],
            evidence[f"{case.case_id}.actual_weights"],
        )
        for name in ("production_direct", "capacity_graph"):
            prefixes[case.case_id][f"{name}_vs_large_reference_routing"] = diagnostics.routing_difference(
                evidence[f"{prefix}.reference_ids"][:row_count],
                evidence[f"{prefix}.reference_weights"][:row_count],
                evidence[f"{case.case_id}.{name}_ids"],
                evidence[f"{case.case_id}.{name}_weights"],
            )
    summary_staging_path = artifact_dir / "gpt-oss-router.json.partial"
    summary_staging_path.write_text(
        json.dumps({"samples": samples, "batch_prefixes": prefixes}, indent=2) + "\n", encoding="utf-8"
    )
    summary_staging_path.replace(artifact_dir / "gpt-oss-router.json")
    # Collect every layer/shape before retaining the original exact assertions.
    for case, result in zip(cases, results, strict=True):
        assert result.router_logits is not None and result.routing is not None
        # Exact BF16 logits are required before interpreting cutoff/tie differences.
        torch.testing.assert_close(evidence[f"{case.case_id}.actual_logits"], result.router_logits, rtol=0, atol=0)
        torch.testing.assert_close(evidence[f"{case.case_id}.actual_ids"], result.routing.topk_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            evidence[f"{case.case_id}.actual_weights"], result.routing.topk_weights, rtol=0, atol=0
        )
