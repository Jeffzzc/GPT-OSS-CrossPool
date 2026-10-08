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

The exact Router test runs a second, Router-only child with the production
workspace policy installed before any CUDA initialization. It reads only the
checkpoint's Router weight/bias tensors and shared hidden-state prefixes. It
captures the existing production Router operator at the derived graph
capacities: rows 31/32 use capacity 32 and rows 33 use capacity 64. All buffers
are allocated before capture; CPU copies and diagnostic profiling occur after
replay. Its raw logits, IDs, weights, tensor geometry, environment and kernel
traces remain separate from the inherited-environment reference evidence.

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
