# GPT-OSS validation contract

## Profile and reference

The qualification identity is `openai/gpt-oss-20b`, architecture
`GptOssForCausalLM`. Require BF16 payloads, attention TP=1 and DP=1,
FFN TP=1 or TP=2 for the serving qualification scenes, and
`--disable-hybrid-swa-memory`. This keeps sliding attention math in SGLang
while using CrossPool's existing full-storage elastic KV pool.

Use the configured checkpoint exclusively through
`XpoolConfig.model_path_of(ModelId("openai/gpt-oss-20b"))`. The adapter requires
24 sparse layers, hidden/intermediate width 2880, 32 Experts, TopK=4,
`hidden_act="silu"`, alpha=1.702, limit=7 and official MXFP4 exclusions.
Checkpoint headers and source tensors are validated on the executing host.
Remote metadata is insufficient evidence for the local checkpoint.

The source reference is pinned [SGLang 0.5.20 GPT-OSS](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/gpt_oss.py)
and its [MXFP4 implementation](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/quantization/mxfp4.py).
Numerical reference jobs disable CrossPool plugins and load the original model
with `moe_runner_backend="triton_kernel"` and `dtype="bfloat16"`. The backend
value matches `MoeRunnerBackend.TRITON_KERNELS.value` in SGLang 0.5.20;
`triton_kernels` remains the Python package name. They record
actual Router logits and invert the original Expert-major TopK carrier into
token/slot order. CrossPool operators do not produce reference tensors.

## Expected checkpoint resources

For each prefix `model.layers.L.mlp`, enforce these source tensors:

| Suffix | Shape | Dtype |
| --- | --- | --- |
| `router.weight` | `[32, 2880]` | BF16 |
| `router.bias` | `[32]` | BF16 |
| `experts.gate_up_proj_blocks` | `[32, 5760, 90, 16]` | uint8 |
| `experts.down_proj_blocks` | `[32, 2880, 90, 16]` | uint8 |
| `experts.gate_up_proj_scales` | `[32, 5760, 90]` | uint8 |
| `experts.down_proj_scales` | `[32, 2880, 90]` | uint8 |
| `experts.gate_up_proj_bias` | `[32, 5760]` | BF16 |
| `experts.down_proj_bias` | `[32, 2880]` | BF16 |

Each scale covers 32 reduction elements. Reject reserved UE8M0 NaN bytes.
Flatten only the final packed block dimension, project only the current
TP shard, and retain packed weights, scales and biases as six independent
device storages. The Triton W4A16 path decodes tiles into registers and uses
BF16 operands with FP32 accumulators. It does not claim hardware FP4 GEMM.

Derive local width as `ceil(90/TP)*32`. TP=2 gives 1440; TP=4 gives 736 with
672 valid elements on the final rank. The TP=4 synthetic memory world tests
padding geometry; it does not qualify an additional serving topology. Empty
final partitions fail admission. Packed tails and W13 bias tails are zero,
tail scales encode one, and W2 bias is zero on TP followers.

## Numerical and graph verdicts

The model suite covers layers `(0, 12, 23)` and rows `(1, 31, 32, 33, 4096)`.
Keep the existing model-owned output and routing tolerances. The additional
checkpoint Router test requires exact BF16 logits, TopK IDs and BF16-rounded
weights and writes `gpt-oss-router.safetensors` before parity assertions.
Its [diagnostic contract](router-parity.md) records GEMM environments and
capacity probes separately and reserves cleanup time within the explicit
1800-second test deadline. Numerical acceptance remains unchanged.

Diagnose failures in this order: checkpoint decode, local TP projection,
Router logits, TopK, W13 accumulator/bias, clamped activation, W2/bias,
route combine/TP reduction, Graph replay and per-layer resource rebinding.
The activation test calls the original pinned `swiglu_fn`, including the
asymmetric gate clamp, symmetric up clamp, FTZ exponential and fused
multiply-add. W13 remains FP32 through that epilogue; activation and weighted
W2 route outputs round to BF16 before combine.

Component tests require no steady-execution Tensor allocation. Native tests
exercise Primary/Control schema validation and device updates of all eight
actual GPT-OSS resources across distinct layers. Resource accounting includes
the six packed storages, Router weight/bias, FP32 W13 workspace, control probes
and the expanded binding table.

Serving compares EAGER with DECODE_FULL for decode token identity and with
DECODE_FULL_PREFILL_BREAKABLE for first-prefill logits/KL. Graph Observer
evidence must prove the requested graph structure and actual Breakable use.
Timeout, skip, xfail and tolerance relaxation cannot replace acceptance.

## Server commands and invalidation

For a reference backend configuration fix, first verify the pinned enum and
run the focused reference-harness tests before model qualification:

```bash
uv run python - <<'PY'
from sglang.srt.layers.moe.utils import MoeRunnerBackend
assert MoeRunnerBackend.TRITON_KERNELS.value == "triton_kernel"
print(MoeRunnerBackend.TRITON_KERNELS.value)
PY
uv run pytest tests/suites/unit/xtest/harness/sglang/reference/test_ffn.py -v
```

Before qualification, verify MPS endpoint ownership for the selected attention
device. An existing endpoint is an ownership conflict until its owner is
identified and retired through the normal lifecycle. Preserve foreign endpoints
and the fail-closed ownership checks. A startup ownership failure provides no
FFN numerical or Graph replay verdict.

Follow the [build requirements](../../../README.md#requirements),
[installation instructions](../../tutorials/quick-start.md) and
[test environment conventions](../../../tests/README.md#commands).
Build the native module and generated stubs at ABI 85 before type checks or
tests. Recompute memory calibration because ABI, spec identities, workspaces
and the packed calibration corpus changed. The corpus includes fitted C4
and held-out H0 packed worlds. Resource qualification must require zero
underprediction at every real-model startup stage, including memory-pressure
startup, as defined in [Qualification](../../designs/qualification.md).
Run `uv run clang-format --dry-run --Werror` with the explicit native paths
in the file scope below, and retain the native documentation check from the
repository's normal quality workflow.

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
uv run xpool memory-profile
uv run ruff format --check src tests
uv run ruff check src tests
uv run ty check
uv run xtest run --suite unit --suite cext --suite integration --strict-requirements
uv run xtest run --integration=sglang --suite openai/gpt-oss-20b --strict-requirements
uv run xtest run --strict-requirements
```

The runtime, operator, native resource and memory changes invalidate affected
existing numerical, graph and resource evidence. Requalify the existing MoE
models and the other affected model suites on the same frozen build:

```bash
uv run xtest run --integration=sglang --suite Qwen/Qwen3-30B-A3B --strict-requirements
uv run xtest run --integration=sglang --suite deepseek-ai/DeepSeek-V2-Lite-Chat --strict-requirements
uv run xtest run --integration=sglang --suite zai-org/GLM-4.7-Flash --strict-requirements
```

Update `docs/supported-models.md` for 20B only after required evidence passes.
GPT-OSS-120B remains outside this qualification and is rejected by the profile.
After acceptance, fold the durable contract into its current design owners
using `write-design`; retain the scoped plan until confirmed cleanup.

## File scope

Paths are relative to the repository root. Existing model-family mathematics
and third-party sources remain owned by their original implementations.

| File | Reason |
| --- | --- |
| `docs/plans/gpt-oss-mxfp4/README.md` | Scoped target design and dependency boundary. |
| `docs/plans/gpt-oss-mxfp4/validation.md` | Checkpoint invariants, evidence and server acceptance instructions. |
| `configs/deployments/openai%2Fgpt-oss-20b/atn1-ffn1-lanes1.toml` | Portable attention TP1 / FFN TP1 serving scene. |
| `configs/deployments/openai%2Fgpt-oss-20b/atn1-ffn2-lanes1.toml` | Portable attention TP1 / FFN TP2 serving scene. |
| `src/xpool/ffn.py` | Packed Expert keys/kind, additive Router bias, clamped activation and shared TP geometry. |
| `src/xpool/runtime/ffnagent/models/gpt_oss.py` | Strict architecture compiler and exact routing semantics. |
| `src/xpool/integrations/sglang/models/gpt_oss.py` | Parameter-free replacement, specific loader filter and fail-closed serving boundary. |
| `src/xpool/runtime/ffnagent/weights.py` | Six-storage packed owner and independent Router projection bias. |
| `src/xpool/runtime/ffnagent/loader.py` | Strict source shape/dtype checks, bounded packed shard staging, tail padding and bias ownership. |
| `src/xpool/runtime/ffnagent/operators.py` | Caller-owned biased routing, selected softmax, packed W4A16 GEMMs and precise activation. |
| `src/xpool/runtime/ffnagent/execution.py` | Distinct execution signatures, FP32 epilogue workspace and exact packed byte geometry. |
| `src/xpool/runtime/ffnagent/registry.py` | Packed capture/control probes, typed bindings and per-layer validation. |
| `src/xpool/runtime/ffnagent/device_memory.py` | Packed resident bytes, Router bias and block-aligned local geometry. |
| `src/xpool/runtime/ffnagent/memory_profile/corpus.py` | Packed fitted and held-out calibration worlds and synthetic weight materialization. |
| `src/xpool/service/daemon/ffn_placement.py` | Use the same TP geometry in daemon-authored Plans. |
| `src/cext-include/xpool/ffnagent/projection.hpp` | Explicit projection-bias and packed auxiliary resource schema. |
| `src/cext-include/xpool/ffnagent/runtime.cuh` | Additional device binding-table addresses. |
| `src/cext/ffnagent/projection.cpp` | Primary/Control and target schema/resource discovery validation. |
| `src/cext/ffnagent/runtime.cu` | Populate, validate and rebind every additional address in Lane Graphs. |
| `src/cext-bindings/fabric.cpp` | Bind the new native projection resources with backward-compatible defaults. |
| `src/cext-include/xpool/abi.hpp` | Increment native ABI to 85. |
| `src/xpool/cext.py` | Require ABI 85 in the Python loader. |
| `src/xpool-dev/xkit/serving/sglang/launch.py` | Explicit BF16 and full-storage SWA launch policy. |
| `src/xpool-dev/xkit/serving/sglang/server.py` | Preserve those policies in SGLang CLI arguments. |
| `src/xpool-dev/xtest/harness/sglang/catalog.py` | Typed numerical reference and serving launch policies. |
| `src/xpool-dev/xtest/harness/sglang/numerical.py` | Reference policies and additional representative middle-layer coverage. |
| `src/xpool-dev/xtest/harness/sglang/reference/ffn_protocol.py` | Carry dtype/backend through isolated reference jobs. |
| `src/xpool-dev/xtest/harness/sglang/reference/ffn.py` | Validate reference policies and read optional Router-logit evidence. |
| `src/xpool-dev/xtest/harness/sglang/reference/ffn_child.py` | Load original GPT-OSS, record actual logits and invert ragged routing carriers. |
| `src/xpool-dev/xtest/harness/sglang/serving/launch.py` | Forward the case-owned launch policies. |
| `tests/suites/models/openai/gpt-oss-20b/test_sglang_model_qualification.py` | Actual-checkpoint numerical/routing evidence and all three graph modes for FFN TP1/TP2. |
| `tests/suites/unit/runtime/ffnagent/models/gpt_oss/test_architecture.py` | Strict profile, spec roundtrip, activation identity, block TP and workspace coverage. |
| `tests/suites/integration/runtime/ffnagent/test_mxfp4_loader.py` | Independent packed slicing, invalid dtype/shape/scale, padding and bias ownership. |
| `tests/suites/integration/runtime/ffnagent/test_mxfp4_operators.py` | Independent decode/GEMM math, original activation, replay and allocation checks. |
| `tests/suites/integration/runtime/ffnagent/models/gpt_oss/test_router.py` | Exact cutoff/tie, additive bias, capture and allocation checks. |
| `tests/suites/integration/sglang/models/gpt_oss/test_shim.py` | Pinned constructor/loader, parameter-free coverage and topology/dtype admission. |
| `tests/suites/cext/ffnagent/projection_test.cpp` | Complete and malformed packed native resource schemas. |
| `tests/suites/cext/ffnagent/parameterization_test.cu` | Actual CUDA Graph device rebinding of all eight layer resources. |
| `tests/suites/cext/abi_test.cpp` | Assert the revised ABI. |
| `tests/suites/unit/runtime/ffnagent/test_device_memory.py` | Resident/probe byte counts and Plan/Signature geometry agreement. |
| `tests/suites/unit/runtime/ffnagent/test_memory_profile.py` | Packed fitted and held-out corpus coverage. |
| `tests/suites/unit/xkit/serving/sglang/test_server.py` | Explicit BF16/SWA CLI propagation. |
| `tests/suites/unit/xtest/harness/sglang/reference/test_ffn.py` | Reference job policies and Router-logit artifact decoding. |
