"""Official MXFP4 GPT-OSS-20B numerical, routing and serving qualification."""

from __future__ import annotations

from functools import partial
from pathlib import Path

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
from xtest.harness.sglang.reference.ffn import SglangFfnReferenceCase, SglangFfnReferenceRunner
from xtest.harness.sglang.serving import qualification
from xtest.harness.support.ffn import generate_ffn_hidden_states

pytest_plugins = ("xtest.harness.support.config",)

MODEL = ModelId("openai/gpt-oss-20b")
LAYERS = (0, 12, 23)
ROWS = (1, 31, 32, 33, 4096)
NUMERICAL_CASES = (
    E2eFfnNumericalCase(
        description="Original pinned SGLang MXFP4 FFN parity across representative layers and capacity boundaries.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        model_id=MODEL,
        layer_ids=LAYERS,
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=ROWS),
        reference_moe_runner_backend="triton_kernels",
        reference_dtype="bfloat16",
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


@xtest.requirements(device_count=2, requires_config=True, model_ids=(MODEL,))
def test_checkpoint_router_projection_and_topk(
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Locate routing errors using actual logits and routes from isolated original SGLang."""

    model_path = e2e_base_config.config.model_path_of(MODEL)
    architecture.load(model_id=MODEL, model_path=model_path)
    inputs = generate_ffn_hidden_states(hidden_size=2880, seed=17, row_counts=ROWS)
    cases = tuple(
        SglangFfnReferenceCase(case_id=f"layer-{layer}-rows-{rows}", layer_id=layer, hidden_states=values)
        for layer in LAYERS
        for rows, values in zip(ROWS, inputs, strict=True)
    )
    results = SglangFfnReferenceRunner(workdir=tmp_path / "reference", timeout_seconds=1800).run(
        model_path=model_path,
        tensor_parallel_size=2,
        cases=cases,
        moe_runner_backend="triton_kernels",
        dtype="bfloat16",
    )
    key_view = checkpoint.read_checkpoint_key_view(model_path)
    evidence: dict[str, torch.Tensor] = {}
    for case, result in zip(cases, results, strict=True):
        prefix = f"model.layers.{case.layer_id}.mlp.router"
        router_tensors = []
        for suffix in ("weight", "bias"):
            key = f"{prefix}.{suffix}"
            with safe_open(str(key_view[key]), framework="pt", device="cpu") as shard:
                router_tensors.append(shard.get_tensor(key).to(device="cuda"))
        router = weights.MoeRouterWeights(
            weight=router_tensors[0], correction_bias=None, projection_bias=router_tensors[1]
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
        # Preserve the failing stage's evidence before any parity assertion.
        safetensors.torch.save_file(evidence, (task_artifact_dir or tmp_path) / "gpt-oss-router.safetensors")
        # Exact BF16 logits are required before interpreting cutoff/tie differences.
        torch.testing.assert_close(logits, result.router_logits, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu(), result.routing.topk_ids, rtol=0, atol=0)
        torch.testing.assert_close(route_weights.cpu(), result.routing.topk_weights, rtol=0, atol=0)
