# Official GPT-OSS MXFP4 model adaptation

## Goal

Run the configured `openai/gpt-oss-20b` checkpoint through the existing
model-adapter, FFN execution, placement, and parameterized GraphTemplate seams.
Support remains conditional on the model-owned numerical and serving suites;
family discovery does not confer qualification on another model ID.

## Baseline

The existing MoE contract retains floating-point, gate-then-up Expert weights,
has an optional corrected-routing bias, and supports ordinary SiLU. It cannot
express GPT-OSS unchanged. Pinned SGLang 0.5.20, tag commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, uses ordinary Router projection
bias, interleaved gate/up projections, two Expert biases, packed MXFP4 weights
and UE8M0 scales. Its ordinary floating-point fused MoE launcher is not an
MXFP4 launcher. The GPT-OSS reference selects specialized MXFP4 backends.

## Accepted Changes

The implementation scope includes the following capabilities. Model paths
remain owned by `XpoolConfig.model_path_of`; checkpoint validation is mandatory
on the serving host and does not substitute remotely inspected metadata for
local tensor evidence.

- Add a strict `GptOssForCausalLM` compiler for the 20B profile: 24 sparse
  layers, H=2880, I=2880, 32 Experts, TopK=4, MXFP4, alpha=1.702, limit=7.
  Reject other profiles and ambiguous quantization/activation settings.
- Distinguish ordinary Router projection bias from correction bias.
- Add an explicit packed MXFP4 Expert owner with six independently owned
  storages: W13 blocks, W2 blocks, W13 scales, W2 scales, W13 bias, W2 bias.
  The representation is interleaved gate/up and is never a BF16 replica.
- Add the precise clamped SwiGLU semantic:
  `g=min(gate, limit)`, `u=clamp(up, -limit, limit)`,
  `g * sigmoid(alpha*g) * (u+1)`, after W13 bias and before W2.
- Match the pinned `triton_kernels` Router carrier: stable lower-Expert-ID
  tie breaking on BF16 logits, FP32 softmax of selected logits, BF16 weight
  rounding, then exact conversion to the Fabric FP32 carrier. The reference
  suite selects this original MXFP4 implementation with the SGLang 0.5.20
  configuration value `moe_runner_backend="triton_kernel"`. The plural
  `triton_kernels` names the Python package and Router carrier. Other backend
  routing policies are outside this first profile.
- Derive TP width as `ceil((I/32)/TP)*32` everywhere. Zero-fill the tail of
  the last shard. Retain W2 bias only on rank zero before route weighting.
- Extend Python/native binding schemas and the layer-binding table with all
  five additional resource addresses (four Expert resources and Router bias).
  Primary/control address discovery and replay use the existing mechanism.
- Keep sliding/full attention math in SGLang and use the existing elastic
  full-storage KV pool. Reject SGLang's distinct hybrid SWA memory allocator
  before model load; serving cases explicitly pass
  `--disable-hybrid-swa-memory`. The serving launch value carries this Boolean
  policy without changing runtime cache ownership.
- Require attention TP=1 and BF16 payloads explicitly; qualification launches
  pass `--dtype bfloat16`. The official config omits a dtype, so relying on
  SGLang's generic `auto` default can select FP16. Reference jobs retain the
  same explicit dtype policy independently of their Expert backend choice.

## Interface Changes

`ActivationKind` gains clamped SwiGLU; model and execution signatures include
alpha and clamp limit. MoE specs gain an explicit Expert weight kind and packed
checkpoint keys. Old adapters keep floating-point defaults. Router owners and
signatures gain a separate optional projection bias. Native constructors keep
defaults for old resource schemas and validate complete scale/bias pairs.

## Data-Structure Changes

The packed weight owner stores uint8 blocks shaped `[E,2*Ir,H/2]` and
`[E,H,Ir/2]`, uint8 scales `[E,2*Ir,H/32]` and `[E,H,Ir/32]`, and BF16 biases
`[E,2*Ir]` and `[E,H]`. Source blocks have an additional trailing dimension
`[.../32,16]`; bounded checkpoint staging flattens only this dimension and
projects the current TP shard. Source dtypes and shapes are checked exactly.
No complete checkpoint replica is retained. Workspace continues to hold
aligned routes, W13/activation/W2 outputs and Router logits.

## Implementation

Use a CrossPool-owned W4A16 Triton Expert matmul within the existing MoE
operator dispatch. Decode E2M1 nibbles and UE8M0 scales tilewise into registers,
cast decoded operands to BF16, and accumulate GEMMs in FP32. This is a native
packed retained representation, not a claim that the ordinary SGLang Triton
launcher accepts MXFP4. Reuse SGLang's caller-output TopK, route alignment and
combine primitives only through `runtime.ffnagent.operators`. The pinned
Triton `_topk_forward` is called with caller-owned BF16/int16 carriers and
bitmatrix scratch. Kernel selection
is signature-driven; no generic operator contains model IDs. W13 FP32
accumulators and bias pass through the clamped activation before BF16 output
rounding, matching the reference's fused epilogue. The explicit FP32 W13
workspace is included in capture, memory estimation and calibration.

The attention shim matches `GptOssSparseMoeBlock(layer_id, config,
quant_config=None, prefix="")`, retains no Expert parameters and provides the
minimal weight-discovery compatibility surface. Its around-hook removes only
canonical decoder FFN tensors. The original MXFP4 loader is still invoked:
its empty Expert iterator performs no parameter lookup and its normal loader
retains the original attention/norm/embedding mappings.

Resident and control-probe accounting includes packed bytes, scales, biases
and block-aligned TP geometry. Capture uses caller-owned buffers and adds no
runtime tensor allocation. Memory corpus identities include the revised spec
digest and execution signature.
Increment the native ABI to invalidate older bindings and memory calibration.
Include packed MXFP4 in fitted and held-out synthetic memory corpus worlds.

## Validation

Add strict profile, checkpoint-layout, dtype, TP-tail, bias ownership,
activation, resource-schema and replay/rebinding coverage. Check independent
MXFP4 decoding and activation formulas; references must not call CrossPool's
operators. Numerical model evidence covers layers 0, 12, 23 and rows 1, 32,
4096, with focused checks around capacity boundaries. Inspect Router logits,
TopK IDs and weights before interpreting final-output mismatches.

Serving evidence covers EAGER, DECODE_FULL and
DECODE_FULL_PREFILL_BREAKABLE on explicit supported topologies, including
FFN TP=2. Reject unsupported attention/Expert-parallel configurations rather
than falling back. Run affected existing unit/integration/native tests and
`uv run xtest run --integration=sglang --suite openai/gpt-oss-20b --strict-requirements`.
Only a successful required qualification permits changing the support table.
The [validation contract and file scope](validation.md) lists checkpoint
invariants, diagnostic evidence, rebuild/calibration requirements and server
commands for this change.

## Out of Scope

GPT-OSS-120B qualification, other GPT-OSS profiles, alternate serving engines,
checkpoint downloads, changing the pinned SGLang version, and modifications
to third-party implementation files.
