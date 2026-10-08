# Router projection and reference diagnostics

## Scope

Diagnose Router parity before changing production mathematics or qualification
criteria. Preserve all rows `(1, 31, 32, 33, 4096)`, layers `(0, 12, 23)`,
the exact Router assertions and the existing FFN/routing tolerances.

`FfnAgent.__init__` installs `CUBLAS_WORKSPACE_CONFIG=:0:0` before CUDA
initialization. The original-SGLang reference inherits its launch environment.
These are separate workspace-policy owners; compare their recorded values
before attributing a GEMM algorithm or numerical discrepancy to them.
Mutating the variable after CUDA/cuBLAS
initialization does not establish an equivalent experiment.

On A100 the SM90/Blackwell TinyGemm path is ineligible. Inspect the installed
`TinyGemmLinear.forward`, `ReplicatedLinear.forward`, and
`UnquantizedLinearMethod.apply`, then verify the actual dispatch with kernel
profiles. The pinned unquantized implementation can dispatch through its BF16
backend selection before its `F.linear` fallback. Record that selection rather
than assuming every unquantized call uses the same GEMM.

Pinned Torch 2.13 supplies a concrete mechanism to test. Its open-source CUDA
build defaults to unified cuBLAS/Lt workspace. `getCUDABlasLtWorkspaceSize()`
caps the Lt limit at the cuBLAS limit; `:0:0` therefore also limits Lt to zero.
Biased GEMM passes that limit into cuBLASLt algorithm selection. This explains
how a process policy can exclude scratch-using algorithms while leaving
`F.linear` and caller-output `addmm` equivalent within each environment.
See the pinned [workspace owner](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/cuda/CublasHandlePool.cpp)
and [biased GEMM](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/cuda/CUDABlas.cpp).
Confirm the installed effective byte limits and kernel traces before concluding
that this mechanism explains every measured discrepancy.

## Evidence contract

`SglangFfnReferenceRunner.run(diagnostics=True)` opts into rank-local JSONL
progress, UTC timestamps, monotonic timestamps, process/rank identities and
Python stack dumps every 120 seconds. Entry records precede device binding,
configuration, distributed initialization, group setup, model loading, each
FFN case, barriers and cleanup. Each JSONL append closes its file, preserving
the completed prefix if the child is cancelled or killed. Signal handlers and
collective timeouts remain unchanged. Additional GPU work and synchronization
are confined to diagnostic probes outside production capture/replay.

The entire original FFN batch completes and is saved before diagnostic Router
calls can warm BLAS/allocator caches or initialize profiler state. Rank zero also
saves the installed Router/quant-method source, BF16 GEMM backend, input and
parameter strides/alignment, raw original Router logits, `F.linear` and
caller-output `addmm` logits, and a Chrome kernel trace. These probes reuse
the exact loaded checkpoint Router tensors and input prefixes. Diagnostic
model cases retain the reference directory under the task artifact directory.
The FFN numerical harness additionally joins reference logits/routes and
observed production routes into `ffn-router-diagnostics.safetensors` and JSON.

The exact Router test runs two fresh Router-only children, explicitly unsetting
`CUBLAS_WORKSPACE_CONFIG` in one and installing `:0:0` in the other before any
CUDA initialization. They read identical checkpoint Router weight/bias tensors
and hidden-state prefixes. Other GEMM controls remain inherited and unchanged.
Each child evaluates the production Router at exact-row eager and exact-row
Graph shapes, the selected Graph capacity, and both eager and Graph shapes at
capacities 32 and 64 whenever they contain the live rows. Rows 4096 are retained
as the shared-prefix control. All buffers are allocated before capture; CPU
copies and profiling occur outside capture.

`gpt-oss-router-workspace-ab.json` separates workspace differences at identical
geometry, shape differences under one policy, and eager/Graph differences at
identical geometry. It also compares each policy/variant to the saved original
SGLang reference. Each policy directory retains raw logits, IDs, weights,
geometry, effective cuBLAS/Lt workspace byte limits, kernel launch inventories
and Chrome traces. No policy setting is changed in the original reference.
The original reference queries effective workspace limits only after its entire
FFN batch is saved, because Torch caches environment-derived defaults when
those getters first run. Fresh-policy probes query them after installing their
policy and before their first GEMM.

This isolated capacity probe is not a live FfnAgent Lane Graph observation.
The existing Routing Observer supplies actual production IDs/weights; it does
not expose Router logits. Workspace addresses and alignment in the probe belong
to its own allocations. Adding live workspace observations would require a
separate Devkit/resource-lifetime design, not a pointer-based workaround.

`gpt-oss-router.json` counts differing BF16 elements, reports maximum absolute
errors and up to eight first mismatches, preserves ordered and sorted Expert
IDs, and associates weights by Expert ID for matching sets. It compares every
small batch to the identical prefix of batch 4096 and reconstructs selected
softmax weights from saved logits. All samples are saved before the original
exact assertions run. None of these diagnostic summaries changes acceptance.

The environment snapshots include Torch/CUDA/package versions, capability,
BLAS preference, reduced-precision controls, workspace settings and selected
NCCL diagnostic settings. The child never changes precision flags or NCCL
settings to obtain a passing result.

## Deadlines and isolation

The Router test declares a pytest timeout of 1800 seconds and an estimated
duration of 900 seconds. xtest collects that timeout into its execution task;
it does not retain the repository default of 300 seconds for this item.
Reference loading and both projection environments share an absolute deadline
120 seconds shorter than the test timeout. This reserve includes the existing
Python child TERM/KILL limits of 30 seconds each, plus test-side finalization.
Supervisor-managed cleanup retains its separate existing lifecycle deadline.

Run the exact Router test on two reserved devices without concurrent serving
work. First inspect its rank progress and stack logs. Only if a rank still
stalls, repeat the bounded test with explicit `NCCL_DEBUG=INFO`,
`NCCL_DEBUG_SUBSYS=INIT,COLL` and, when needed,
`TORCH_DISTRIBUTED_DEBUG=DETAIL`. Identify the last entered operation and
missing rank completion before proposing any distributed-runtime change.

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
uv run pytest tests/suites/unit/xtest/harness/sglang/reference -v
uv run xtest run --suite openai/gpt-oss-20b -k test_checkpoint_router_projection_and_topk --strict-requirements
uv run pytest tests/suites/integration/runtime/ffnagent/models/gpt_oss/test_router.py -v --strict-requirements
uv run xtest run --integration=sglang --suite openai/gpt-oss-20b --strict-requirements
```

## Decision boundary

Attribute a discrepancy to projection only after comparing saved logits in
both fresh-process environments. Equal logits with different routes require
TopK diagnosis instead. Compare exact-shape eager execution to capacity-shape
replay independently of the workspace experiment; the two variables must not
be conflated.

If matching GEMM environments fixes exact-shape parity but capacity replay
still differs, retain the actual-row-count qualification failure. A proposal
to use capacity-matched reference projections must include per-layer/per-row
evidence and an explicit review of the changed numerical-compatibility claim.
Alternatively, an actual-row-count projection during fixed-capacity execution
requires an accepted operator/graph design. Neither alternative is implemented
or accepted by this diagnostic change. Expert MXFP4 execution, TP slicing,
W13/W2 and serving Graph resource schemas remain outside its scope.

## Controlled result and production repair

Controlled A100 experiments established workspace policy as the cause of the
nine discrepancies: unset matched all 15 original-SGLang samples exactly;
`:0:0` matched six and reproduced the nine small-batch failures. Actual-row
and capacity execution, eager and Graph replay, were identical under each
policy. Expert sets remained equal. The maxima below are over rows 31/32/33,
not claims about each individual row count or validation of the repair.

| Layer | Affected rows | Maximum BF16 logit error | Maximum routing weight error |
| --- | --- | --- | --- |
| 0 | 31, 32, 33 | 0.015625 | 0.005859375 |
| 12 | 31, 32, 33 | 0.00390625 | 0.001953125 |
| 23 | 31, 32, 33 | 0.00390625 | 0.001953125 |

Torch caches environment-derived workspace limits process-wide; handles are
pooled per host thread/device and implicit scratch is keyed by handle/stream,
not by Executor Lane. Standard `addmm(out=...)` supplies an output destination,
not caller-owned Lt scratch. Enlarging that implicit pool cannot establish Lane
ownership or relocation. A custom GEMM would additionally need experimental
proof of Split-K rounding. The scoped repair therefore uses the native biased
GEMM in the [target delta](README.md#sm80-router-gemm-workspace), preserving
the zero process policy and the original reference.

The explicit operator uses BF16 A/B/C/D, FP32 compute and scalar types, alpha=1,
beta=0, column-major views `W.T @ X.T`, and the bias epilogue. It reproduces
Torch's reduction mask, alignment preferences and first heuristic result with
the pinned default 1 MiB Lt limit. Algorithm identifiers are library/version
dependent; retained kernel traces, rather than a hardcoded ID, validate the
selected GEMM and Split-K reduction. Weight/bias discovery must cover both.

The enclosing Router workspace and its scratch offset are 256-byte aligned.
The captured Torch input is also normally 256-byte aligned, whereas the
Fabric payload guarantees only 16 bytes. Query the selected algorithm's
minimum B alignment and fail closed if it exceeds the Fabric guarantee;
do not silently select another algorithm. The native Lane test deliberately
uses input addresses aligned to 16 but not 256 bytes after relocation.
Scratch contributes one MiB plus alignment padding per Router-owner capture
and one MiB plus padding to the maximum per-Lane workspace. TP followers acquire
no Router scratch. It shares the existing workspace lifetime and relocation
range, so no new layer resource field or pointer lifetime is introduced.
The native test poisons capture storage after installation, alternates layer
weight/bias addresses from a device update node, and launches two independently
relocated Lane Graphs before waiting for either. Synthetic Python tests check
changed-input replay, exact routes and absence of replay allocations.

The exact model test asserts logits, IDs and weights against unchanged SGLang
for every eager/Graph/capacity variant in both policy children, after retaining
all evidence. The original FFN and serving qualification remain required.
Rebuild generated stubs, recalibrate memory and run these tests on SM80 before
claiming the repair is qualified. No threshold or reference-policy change is
part of this repair. If a library version selects a different algorithm or its
captured parameters escape declared resources, preserve the failure and its
trace rather than falling back to unowned scratch.

## Router repair file scope

| File | Target function or contract |
| --- | --- |
| `src/cext-include/xpool/ffnagent/router.hpp` | Declare `biased_router_gemm`, the 1 MiB budget and 256-byte alignment/lifetime contract. |
| `src/cext/ffnagent/router.cpp` | Implement tensor validation, Torch-compatible Lt descriptors/heuristic, explicit scratch and Fabric input-alignment admission. |
| `src/cext/CMakeLists.txt` | Link `CUDA::cublasLt` through the existing native core. |
| `src/cext-bindings/fabric.cpp` | Bind the operator and workspace constants under the existing `native.ffnagent` module. |
| `src/xpool/runtime/ffnagent/operators.py` | Select explicit-scratch GEMM on SM80 inside `compute_biased_router_logits`; preserve TinyGemm and other dispatch. |
| `src/xpool/runtime/ffnagent/models/gpt_oss.py` | Derive logits/TopK/scratch spans and pass their disjoint views through `compute_routed_topk`. |
| `src/xpool/runtime/ffnagent/execution.py` | Align the additive-bias Router region in `moe_workspace_layout`; existing capture/Lane ledgers consume its full extent. |
| `src/xpool/runtime/ffnagent/agent.py` | Document the retained zero-workspace process policy at initialization. |
| `src/cext-include/xpool/abi.hpp` | Advance native identity to 86. |
| `src/xpool/cext.py` | Require ABI 86 and reject stale binaries. |
| `tests/suites/cext/abi_test.cpp` | Require the revised identity. |
| `tests/suites/cext/ffnagent/router_test.cu` | Exercise actual Lt capture schemas, device weight/bias rebinding, 16-byte Lane input and independent relocated scratch. |
| `tests/suites/integration/runtime/ffnagent/models/gpt_oss/test_router.py` | Supply scratch, check repeated changed-input eager/replay/capacity results, peak allocations and invalid scratch rejection. |
| `tests/suites/unit/runtime/ffnagent/models/gpt_oss/test_architecture.py` | Check per-capacity workspace extent/alignment and capture accounting. |
| `tests/suites/unit/runtime/ffnagent/test_device_memory.py` | Prove scratch enters every Router-owner capture and Lane ledger, preserves allocation counts and leaves followers unchanged. |
| `src/xpool-dev/xtest/harness/sglang/reference/router_probe.py` | Record explicit scratch geometry/budget in existing opt-in evidence. |
| `tests/suites/models/openai/gpt-oss-20b/test_sglang_model_qualification.py` | Require exact logits/IDs/weights for all variants under both fresh workspace policies against unchanged SGLang. |
| `docs/plans/gpt-oss-mxfp4/README.md` | Own the explicit-workspace target delta and ownership decisions. |
| `docs/plans/gpt-oss-mxfp4/router-parity.md` | Own causal evidence, numerical boundaries and the repair scope. |
| `docs/plans/gpt-oss-mxfp4/validation.md` | Require rebuild, calibration, focused checks and complete qualification. |
