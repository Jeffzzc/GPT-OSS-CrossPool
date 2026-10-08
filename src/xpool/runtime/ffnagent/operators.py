"""Concrete FfnAgent-owned FFN operator composition."""

from __future__ import annotations

import contextlib
import math
import types
import typing
from collections.abc import Generator

import torch
import triton
from triton import language

from xpool.ffn import ActivationKind
from xpool.native import ffnagent
from xpool.runtime.ffnagent import execution, weights
from xpool.utils import align_up

ROUTING_FINALIZATION_BLOCK_SIZE = 256


@triton.jit
def convert_bf16_topk_carrier_kernel(
    ids_pointer,
    weights_pointer,
    output_ids_pointer,
    output_weights_pointer,
    ELEMENT_COUNT: language.constexpr,
    BLOCK_SIZE: language.constexpr,
):
    """Preserve the reference's BF16 weight rounding in the FP32 Fabric carrier."""

    offsets = language.program_id(0) * BLOCK_SIZE + language.arange(0, BLOCK_SIZE)
    mask = offsets < ELEMENT_COUNT
    ids = language.load(ids_pointer + offsets, mask, other=0)
    values = language.load(weights_pointer + offsets, mask, other=0)
    language.store(output_ids_pointer + offsets, ids.to(language.int32), mask)
    language.store(output_weights_pointer + offsets, values.to(language.float32), mask)


def compute_bf16_selected_softmax_topk(
    *,
    logits: torch.Tensor,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """Reuse pinned Triton TopK with caller-owned carriers and bitmatrix.

    This backend selects raw BF16 logits with stable lower-ID tie breaking,
    applies FP32 softmax only to selected logits, then rounds weights to BF16.
    All temporary storage is supplied by the Router workspace.
    """

    from triton_kernels.topk_details._topk_forward import _topk_forward

    rows, expert_count = logits.shape
    topk = routed_ids.shape[1]
    if logits.dtype is not torch.bfloat16 or expert_count != 32 or topk != 4:
        raise ValueError("selected BF16 TopK supports the admitted 32-Expert TopK-4 geometry")
    carrier_bytes = align_up(rows * topk * 2, 16)
    padded_rows = align_up(rows, 32)
    if workspace.numel() != 2 * carrier_bytes + padded_rows * 4:
        raise ValueError("selected BF16 TopK workspace has the wrong extent")
    values = workspace[: rows * topk * 2].view(torch.bfloat16).view(rows, topk)
    ids = workspace[carrier_bytes : carrier_bytes + rows * topk * 2].view(torch.int16).view(rows, topk)
    bitmatrix = workspace[2 * carrier_bytes :].view(torch.uint32)
    _topk_forward[(triton.cdiv(rows, 32),)](
        logits,
        logits.stride(0),
        (values,),
        (ids,),
        topk,
        False,
        (bitmatrix,),
        1,
        padded_rows,
        rows,
        expert_count,
        0,
        APPLY_SOFTMAX=True,
        BLOCK_M=32,
        BLOCK_N=32,
        N_EXPTS_PAD=32,
        N_EXPTS_ACT=4,
    )
    convert_bf16_topk_carrier_kernel[(triton.cdiv(rows * topk, 256),)](
        ids,
        values,
        routed_ids,
        routed_weights,
        ELEMENT_COUNT=typing.cast(language.constexpr, rows * topk),
        BLOCK_SIZE=typing.cast(language.constexpr, 256),
    )


def compute_biased_router_logits(
    *,
    hidden_states: torch.Tensor,
    router_weights: weights.MoeRouterWeights,
    logits: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """Match the pinned BF16 Router projection into caller-owned logits.

    The reference uses FlashInfer TinyGemm for small SM90/Blackwell batches
    and a biased linear GEMM otherwise. Bias is accumulated before BF16 output
    rounding; adding bias to an already rounded unbiased GEMM is different.
    SM80 uses explicit caller-owned Lt scratch, retaining the reference's
    default heuristic budget independently of the process zero-workspace policy.
    """

    from sglang.srt.utils import is_flashinfer_available

    capability = torch.cuda.get_device_capability(hidden_states.device)
    if (
        hidden_states.shape[0] <= 128
        and is_flashinfer_available()
        and (capability == (9, 0) or capability[0] in (10, 12))
    ):
        try:
            from flashinfer.gemm import tinygemm_bf16
        except ImportError:
            tinygemm_bf16 = None
        if tinygemm_bf16 is not None:
            tinygemm_bf16(hidden_states, router_weights.weight, logits, router_weights.projection_bias, use_pdl=False)
            return
    if router_weights.projection_bias is None:
        raise ValueError("biased Router projection requires ordinary projection bias")
    if capability == (8, 0):
        ffnagent.biased_router_gemm(
            hidden_states, router_weights.weight, router_weights.projection_bias, logits, workspace
        )
        return
    torch.addmm(router_weights.projection_bias, hidden_states, router_weights.weight.t(), out=logits)


@triton.jit
def mxfp4_expert_matmul_kernel(
    input_pointer,
    blocks_pointer,
    scales_pointer,
    bias_pointer,
    output_pointer,
    route_weights_pointer,
    sorted_ids_pointer,
    expert_ids_pointer,
    padded_count_pointer,
    N: language.constexpr,
    K: language.constexpr,
    EXPERT_COUNT: language.constexpr,
    ROUTE_COUNT: language.constexpr,
    INPUT_TOPK: language.constexpr,
    MULTIPLY_ROUTE: language.constexpr,
    BLOCK_M: language.constexpr,
    BLOCK_N: language.constexpr,
    BLOCK_K: language.constexpr,
):
    """W4A16 GEMM: decode E2M1/UE8M0 tiles without a dequantized weight allocation."""

    block_m = language.program_id(0)
    if block_m * BLOCK_M >= language.load(padded_count_pointer):
        return
    routes = language.load(sorted_ids_pointer + block_m * BLOCK_M + language.arange(0, BLOCK_M))
    columns = language.program_id(1) * BLOCK_N + language.arange(0, BLOCK_N)
    expert = language.load(expert_ids_pointer + block_m)
    output_offsets = routes[:, None] * N + columns[None, :]
    output_mask = (routes[:, None] < ROUTE_COUNT) & (columns[None, :] < N)
    if expert < 0 or expert >= EXPERT_COUNT:
        language.store(output_pointer + output_offsets, 0.0, output_mask)
        return
    rows = routes // INPUT_TOPK
    reductions = language.arange(0, BLOCK_K)
    accumulator = language.full((BLOCK_M, BLOCK_N), 0, language.float32)
    for block_k in range(language.cdiv(K, BLOCK_K)):
        ks = block_k * BLOCK_K + reductions
        a = language.load(
            input_pointer + rows[:, None] * K + ks[None, :],
            (routes[:, None] < ROUTE_COUNT) & (ks[None, :] < K),
            other=0,
        )
        packed = language.load(
            blocks_pointer + expert * N * (K // 2) + columns[None, :] * (K // 2) + ks[:, None] // 2,
            (columns[None, :] < N) & (ks[:, None] < K),
            other=0,
        ).to(language.int32)
        nibble = (packed >> ((ks[:, None] % 2) * 4)) & 15
        magnitude = nibble & 7
        value = language.where(
            magnitude < 4,
            magnitude.to(language.float32) * 0.5,
            language.exp2((magnitude // 2 - 1).to(language.float32))
            * (1.0 + (magnitude % 2).to(language.float32) * 0.5),
        )
        value = language.where((nibble & 8) != 0, -value, value)
        exponent = language.load(
            scales_pointer + expert * N * (K // 32) + columns[None, :] * (K // 32) + ks[:, None] // 32,
            (columns[None, :] < N) & (ks[:, None] < K),
            other=127,
        ).to(language.int32)
        b = (value * language.exp2((exponent - 127).to(language.float32))).to(language.bfloat16)
        accumulator = language.dot(a, b, accumulator)
    bias = language.load(bias_pointer + expert * N + columns, columns < N, other=0).to(language.float32)
    result = accumulator + bias[None, :]
    if MULTIPLY_ROUTE:
        route_weights = language.load(route_weights_pointer + routes, routes < ROUTE_COUNT, other=0)
        result *= route_weights[:, None]
    language.store(output_pointer + output_offsets, result, output_mask)


@triton.jit
def clamped_swiglu_kernel(
    gate_up_pointer,
    output_pointer,
    ELEMENT_COUNT: language.constexpr,
    ALPHA: language.constexpr,
    LIMIT: language.constexpr,
    BLOCK_SIZE: language.constexpr,
):
    """Activate biased interleaved gate/up values using the exact GPT-OSS formula."""

    offsets = language.program_id(0) * BLOCK_SIZE + language.arange(0, BLOCK_SIZE)
    mask = offsets < ELEMENT_COUNT
    gate = language.load(gate_up_pointer + 2 * offsets, mask, other=0).to(language.float32)
    up = language.load(gate_up_pointer + 2 * offsets + 1, mask, other=0).to(language.float32)
    gate = language.minimum(gate, LIMIT)
    up = language.minimum(language.maximum(up, -LIMIT), LIMIT)
    exponent = (-ALPHA * gate) * 1.4426950408889634
    exponential = language.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=r,r",
        [exponent],
        dtype=language.float32,
        is_pure=True,
        pack=1,
    )
    sigmoid_gate = gate / (1.0 + exponential)
    activated = language.fma(sigmoid_gate, up, sigmoid_gate)
    language.store(output_pointer + offsets, activated, mask)


def compute_clamped_swiglu(
    *,
    gate_up: torch.Tensor,
    output: torch.Tensor,
    alpha: float,
    clamp_limit: float,
) -> None:
    """Write clamped interleaved SwiGLU into caller-owned BF16 storage."""

    if not math.isfinite(alpha) or not math.isfinite(clamp_limit) or min(alpha, clamp_limit) <= 0:
        raise ValueError("clamped SwiGLU parameters must be finite and positive")
    if gate_up.shape != (output.shape[0], output.shape[1] * 2):
        raise ValueError("clamped SwiGLU input/output geometry disagrees")
    if gate_up.dtype is not torch.float32 or output.dtype is not torch.bfloat16:
        raise ValueError("clamped SwiGLU requires FP32 accumulators and BF16 output")
    if gate_up.device != output.device or not gate_up.is_contiguous() or not output.is_contiguous():
        raise ValueError("clamped SwiGLU tensors must be contiguous on one device")
    clamped_swiglu_kernel[(triton.cdiv(output.numel(), 256),)](
        gate_up,
        output,
        ELEMENT_COUNT=typing.cast(language.constexpr, output.numel()),
        ALPHA=typing.cast(language.constexpr, alpha),
        LIMIT=typing.cast(language.constexpr, clamp_limit),
        BLOCK_SIZE=typing.cast(language.constexpr, 256),
        enable_fp_fusion=False,
    )


@triton.jit
def finalize_moe_routing_kernel(
    routed_ids_pointer,
    routed_weights_pointer,
    final_ids_pointer,
    final_weights_pointer,
    payload_rows_pointer,
    ROW_CAPACITY: language.constexpr,
    ROUTED_TOPK: language.constexpr,
    EFFECTIVE_TOPK: language.constexpr,
    ROUTED_EXPERT_COUNT: language.constexpr,
    ROUTED_SCALING_FACTOR: language.constexpr,
    BLOCK_SIZE: language.constexpr,
):
    """Write routed-plus-shared live rows and an invalid Capacity tail."""

    offsets = language.program_id(0) * BLOCK_SIZE + language.arange(0, BLOCK_SIZE)
    element_count = ROW_CAPACITY * EFFECTIVE_TOPK
    valid = offsets < element_count
    rows = offsets // EFFECTIVE_TOPK
    slots = offsets % EFFECTIVE_TOPK
    live = rows < language.load(payload_rows_pointer)
    routed = slots < ROUTED_TOPK
    routed_offsets = rows * ROUTED_TOPK + slots
    routed_ids = language.load(
        routed_ids_pointer + routed_offsets,
        mask=valid & live & routed,
        other=-1,
    )
    routed_weights = language.load(
        routed_weights_pointer + routed_offsets,
        mask=valid & live & routed,
        other=0.0,
    )
    expert_ids = language.where(routed, routed_ids, ROUTED_EXPERT_COUNT + slots - ROUTED_TOPK)
    expert_weights = language.where(routed, routed_weights, 1.0 / ROUTED_SCALING_FACTOR)
    language.store(final_ids_pointer + offsets, language.where(live, expert_ids, -1), mask=valid)
    language.store(final_weights_pointer + offsets, language.where(live, expert_weights, 0.0), mask=valid)


@contextlib.contextmanager
def sglang_moe_config_selection() -> Generator[None, None, None]:
    """Supply the pinned MoE selector's sole startup execution value.

    Side Effects:
        Temporarily replaces the selector module's bound
        ``get_exec`` function and restores the exact previous
        function before returning or propagating an exception. The context is
        single-owner startup state and is not safe for concurrent use.
    """

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    previous = fused_moe_triton_config.get_exec
    setattr(
        fused_moe_triton_config,
        "get_exec",
        lambda: types.SimpleNamespace(
            deterministic=types.SimpleNamespace(enable_deterministic_inference=False),
        ),
    )
    try:
        yield
    finally:
        setattr(fused_moe_triton_config, "get_exec", previous)


def compute_softmax_topk(
    *,
    logits: torch.Tensor,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    renormalize: bool,
) -> None:
    """Write SGLang-kernel Softmax TopK results into caller-owned tensors."""

    import sgl_kernel

    sgl_kernel.topk_softmax(routed_weights, routed_ids, logits, renormalize=renormalize)


def copy_moe_kernel_config(config: object, *, name: str) -> dict[str, int]:
    """Copy one selected private-launcher configuration into plain values."""

    if not isinstance(config, dict):
        raise RuntimeError(f"{name} MoE kernel configuration must be a dict")
    copied: dict[str, int] = {}
    for key, value in config.items():
        if key == "USE_TMA":
            continue
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{name} MoE kernel configuration contains an invalid entry")
        copied[key] = value
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"):
        if key not in copied:
            raise RuntimeError(f"{name} MoE kernel configuration is missing {key}")
    return copied


def select_moe_kernel_configs(
    *,
    layer_weights: weights.MoeFfnWeights | weights.Mxfp4MoeFfnWeights,
    row_capacity: int,
    effective_topk: int,
) -> tuple[dict[str, int], dict[str, int] | None]:
    """Select immutable pinned W13 and optional W2 launch configurations.

    Args:
        layer_weights: Canonical local Expert weights whose shapes select the
            pinned configuration tables.
        row_capacity: Positive fixed Graph row Capacity used as selector ``M``.
        effective_topk: Positive routed-plus-shared route width.

    Returns:
        Plain copied W13 configuration and an optional independently selected
        W2 configuration. TMA is unconditionally disabled.

    Raises:
        ValueError: If Capacity or route width is inconsistent with weights.
        RuntimeError: If the pinned selector returns an invalid mapping.

    Side Effects:
        Imports and consults pinned SGLang startup configuration while a
        dedicated execution context is installed. The original context getter
        is restored before return.
    """

    expert_count = (
        layer_weights.gate_up_blocks.shape[0]
        if isinstance(layer_weights, weights.Mxfp4MoeFfnWeights)
        else layer_weights.expert_gate_up_weight.shape[0]
    )
    if row_capacity <= 0 or effective_topk <= 0 or effective_topk > expert_count:
        raise ValueError("MoE kernel selection dimensions are inconsistent")
    if isinstance(layer_weights, weights.Mxfp4MoeFfnWeights):
        return {
            "BLOCK_SIZE_M": 16 if row_capacity <= 32 else 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }, None

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    with sglang_moe_config_selection():
        selected, down = fused_moe_triton_config.try_get_optimal_moe_config(
            layer_weights.expert_gate_up_weight.shape,
            layer_weights.expert_down_weight.shape,
            effective_topk,
            None,
            row_capacity,
            is_marlin=False,
            block_shape=None,
            per_channel_quant=False,
            return_down_config=True,
        )
    w13_config = copy_moe_kernel_config(selected, name="W13")
    down_config = down[0]
    w2_config = None if down_config is None else copy_moe_kernel_config(down_config, name="W2")
    block_size_m = w13_config["BLOCK_SIZE_M"]
    if block_size_m not in execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES:
        raise RuntimeError(f"W13 BLOCK_SIZE_M {block_size_m} is outside the qualified domain")
    if w2_config is not None and w2_config["BLOCK_SIZE_M"] != block_size_m:
        raise RuntimeError("W13 and W2 BLOCK_SIZE_M must match one alignment")
    return w13_config, w2_config


def finalize_moe_routing(
    *,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    final_ids: torch.Tensor,
    final_weights: torch.Tensor,
    payload_rows: torch.Tensor,
    routed_expert_count: int,
    shared_expert_count: int,
    routed_scaling_factor: float,
) -> None:
    """Finalize fixed-Capacity semantic Routing Metadata in one kernel.

    Args:
        routed_ids: Contiguous CUDA int32 Router output shaped
            ``[C, routed_topk]``.
        routed_weights: Contiguous CUDA FP32 Router output with the same shape.
        final_ids: Caller-owned contiguous CUDA int32 destination shaped
            ``[C, effective_topk]``.
        final_weights: Caller-owned contiguous CUDA FP32 destination with the
            same shape as ``final_ids``.
        payload_rows: One-element contiguous CUDA int64 Tensor containing the
            positive live row count.
        routed_expert_count: Positive routed Expert cardinality.
        shared_expert_count: Nonnegative always-selected Shared Expert count.
        routed_scaling_factor: Finite positive scale used to derive reciprocal
            Shared Expert carrier weights.

    Raises:
        ValueError: If shapes, dtypes, devices, counts, or the warmup live-row
            value disagree with the fixed Signature.

    Side Effects:
        Writes complete fixed-Capacity ID and weight destinations on the
        current CUDA stream. It allocates no Tensor or auxiliary workspace.
    """

    weights.validate_tensor(routed_ids, name="routed Expert ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(routed_weights, name="routed Expert weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(final_ids, name="final Expert ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(final_weights, name="final Expert weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(payload_rows, name="payload rows", dtype=torch.int64, dimensions=1)
    row_capacity, routed_topk = routed_ids.shape
    effective_topk = routed_topk + shared_expert_count
    if routed_weights.shape != routed_ids.shape:
        raise ValueError("routed Expert id and weight dimensions disagree")
    if final_ids.shape != (row_capacity, effective_topk) or final_weights.shape != final_ids.shape:
        raise ValueError("final Routing Metadata dimensions disagree")
    if payload_rows.numel() != 1:
        raise ValueError("payload rows must contain exactly one element")
    if routed_expert_count < routed_topk or shared_expert_count < 0:
        raise ValueError("Routing Metadata Expert counts are inconsistent")
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("routed scaling factor must be finite and positive")
    tensors = (routed_ids, routed_weights, final_ids, final_weights, payload_rows)
    if any(tensor.device != routed_ids.device for tensor in tensors):
        raise ValueError("Routing Metadata tensors must share one device")
    if not torch.cuda.is_current_stream_capturing():
        live_rows = int(payload_rows.item())
        if live_rows <= 0 or live_rows > row_capacity:
            raise ValueError("payload rows must be positive and no greater than Capacity")

    element_count = row_capacity * effective_topk
    grid = (triton.cdiv(element_count, ROUTING_FINALIZATION_BLOCK_SIZE),)
    finalize_moe_routing_kernel[grid](
        routed_ids,
        routed_weights,
        final_ids,
        final_weights,
        payload_rows,
        ROW_CAPACITY=typing.cast(language.constexpr, row_capacity),
        ROUTED_TOPK=typing.cast(language.constexpr, routed_topk),
        EFFECTIVE_TOPK=typing.cast(language.constexpr, effective_topk),
        ROUTED_EXPERT_COUNT=typing.cast(language.constexpr, routed_expert_count),
        ROUTED_SCALING_FACTOR=typing.cast(language.constexpr, routed_scaling_factor),
        BLOCK_SIZE=typing.cast(language.constexpr, ROUTING_FINALIZATION_BLOCK_SIZE),
    )


def compute_moe_partial(
    *,
    hidden_states: torch.Tensor,
    layer_weights: weights.MoeFfnWeights | weights.Mxfp4MoeFfnWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    gate_up: torch.Tensor,
    activated: torch.Tensor,
    route_outputs: torch.Tensor,
    output: torch.Tensor,
    w13_config: dict[str, int],
    w2_config: dict[str, int] | None,
    activation: ActivationKind,
    routed_scaling_factor: float,
    activation_alpha: float | None = None,
    activation_clamp_limit: float | None = None,
) -> torch.Tensor:
    """Compute one fixed-Capacity rank-local MoE Partial.

    Args:
        hidden_states: Caller-owned contiguous payload input shaped ``[C, H]``.
        layer_weights: Canonical local routed-then-shared Expert weight shards.
        topk_ids: Final semantic contiguous int32 routes shaped ``[C, K]``.
        topk_weights: Final semantic contiguous FP32 route weights shaped
            ``[C, K]``.
        sorted_token_ids: Caller-owned int32 alignment output.
        expert_ids: Caller-owned int32 aligned-block Expert IDs.
        num_tokens_post_padded: Caller-owned one-element int32 aligned count.
        cumsum_buffer: Caller-owned int32 alignment scratch.
        gate_up: Caller-owned W13 output shaped ``[C * K, 2 * I_r]``;
            MXFP4 retains FP32 accumulators through the activation epilogue.
        activated: Caller-owned payload activation shaped ``[C * K, I_r]``.
        route_outputs: Caller-owned payload W2 output shaped ``[C, K, H]``.
            Its storage may alias ``gate_up`` because their lifetimes do not
            overlap.
        output: Caller-owned contiguous payload Partial shaped ``[C, H]``.
        w13_config: Frozen private-launcher Gate/Up configuration.
        w2_config: Optional frozen Down configuration; ``None`` reuses W13.
        activation: Startup-selected gated activation semantic.
        routed_scaling_factor: Finite positive post-combine scale.

    Returns:
        The exact ``output`` object after every Capacity row is evaluated.

    Raises:
        ValueError: If tensors, configurations, activation, or geometry do not
            match the fixed Expert implementation.
        ImportError: If a pinned operator dependency cannot load.

    Side Effects:
        Writes all caller-owned workspace and output tensors on the current
        CUDA stream. It allocates no Torch output or persistent tensor.
    """

    packed = isinstance(layer_weights, weights.Mxfp4MoeFfnWeights)
    if packed:
        if (
            activation is not ActivationKind.CLAMPED_SWIGLU
            or activation_alpha is None
            or activation_clamp_limit is None
        ):
            raise ValueError("MXFP4 execution requires explicit clamped SwiGLU semantics")
    elif activation is not ActivationKind.SILU or activation_alpha is not None or activation_clamp_limit is not None:
        raise ValueError("floating-point MoE execution requires ordinary SiLU")
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("MoE routed scaling factor must be finite and positive")
    payload_dtype = hidden_states.dtype
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MoE execution requires BF16 or FP16 payloads")
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(topk_ids, name="topk ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(topk_weights, name="topk weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(sorted_token_ids, name="sorted token ids", dtype=torch.int32, dimensions=1)
    weights.validate_tensor(expert_ids, name="aligned Expert ids", dtype=torch.int32, dimensions=1)
    weights.validate_tensor(
        num_tokens_post_padded,
        name="post-padding token count",
        dtype=torch.int32,
        dimensions=1,
    )
    weights.validate_tensor(cumsum_buffer, name="alignment cumsum", dtype=torch.int32, dimensions=1)
    gate_up_dtype = torch.float32 if packed else payload_dtype
    weights.validate_tensor(gate_up, name="Expert gate/up output", dtype=gate_up_dtype, dimensions=2)
    weights.validate_tensor(activated, name="Expert activation output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(route_outputs, name="Expert route outputs", dtype=payload_dtype, dimensions=3)
    weights.validate_tensor(output, name="MoE Partial output", dtype=payload_dtype, dimensions=2)

    row_capacity, hidden_size = hidden_states.shape
    if isinstance(layer_weights, weights.Mxfp4MoeFfnWeights):
        expert_count, local_intermediate_twice, packed_hidden_size = layer_weights.gate_up_blocks.shape
        weight_hidden_size = packed_hidden_size * 2
        weight_dtype = layer_weights.gate_up_bias.dtype
        expert_tensors = layer_weights.resources()
    else:
        expert_count, local_intermediate_twice, weight_hidden_size = layer_weights.expert_gate_up_weight.shape
        weight_dtype = layer_weights.expert_gate_up_weight.dtype
        expert_tensors = (layer_weights.expert_gate_up_weight, layer_weights.expert_down_weight)
    local_intermediate_size = local_intermediate_twice // 2
    if weight_hidden_size != hidden_size or weight_dtype is not payload_dtype:
        raise ValueError("MoE input and Expert hidden dimensions disagree")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != row_capacity:
        raise ValueError("MoE TopK id and weight dimensions disagree")
    effective_topk = topk_ids.shape[1]
    route_count = row_capacity * effective_topk
    if gate_up.shape != (route_count, local_intermediate_twice):
        raise ValueError("MoE Gate/Up workspace dimensions disagree")
    if activated.shape != (route_count, local_intermediate_size):
        raise ValueError("MoE activation workspace dimensions disagree")
    if route_outputs.shape != (row_capacity, effective_topk, hidden_size):
        raise ValueError("MoE route-output workspace dimensions disagree")
    if output.shape != hidden_states.shape:
        raise ValueError("MoE Partial output dimensions disagree")

    w13 = copy_moe_kernel_config(w13_config, name="W13")
    w2 = w13 if w2_config is None else copy_moe_kernel_config(w2_config, name="W2")
    block_size_m = w13["BLOCK_SIZE_M"]
    if w2["BLOCK_SIZE_M"] != block_size_m:
        raise ValueError("W13 and W2 BLOCK_SIZE_M must match one alignment")
    maximum_padded, expert_block_count, cumsum_count = execution.moe_alignment_workspace_shapes(
        row_capacity=row_capacity,
        effective_topk=effective_topk,
        expert_count=expert_count,
        block_size_m=block_size_m,
    )
    if sorted_token_ids.numel() != maximum_padded:
        raise ValueError("sorted-token workspace has the wrong element count")
    if expert_ids.numel() != expert_block_count:
        raise ValueError("aligned Expert-id workspace has the wrong element count")
    if num_tokens_post_padded.numel() != 1 or cumsum_buffer.numel() != cumsum_count:
        raise ValueError("MoE alignment scalar or cumsum workspace has the wrong element count")

    tensors = (
        hidden_states,
        *expert_tensors,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        gate_up,
        activated,
        route_outputs,
        output,
    )
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("MoE Expert tensors must share one device")

    import sgl_kernel

    output_dtype = language.bfloat16 if payload_dtype is torch.bfloat16 else language.float16
    # The pinned extension writes the three caller-owned alignment buffers;
    # the final flag routes invalid Capacity padding to the sentinel Expert.
    sgl_kernel.moe_align_block_size(
        topk_ids,
        expert_count + 1,
        block_size_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    if isinstance(layer_weights, weights.Mxfp4MoeFfnWeights):
        for inputs, blocks, scales, bias, destination, width, reduction, input_topk, multiply in (
            (
                hidden_states,
                layer_weights.gate_up_blocks,
                layer_weights.gate_up_scales,
                layer_weights.gate_up_bias,
                gate_up,
                local_intermediate_twice,
                hidden_size,
                effective_topk,
                False,
            ),
            (
                activated,
                layer_weights.down_blocks,
                layer_weights.down_scales,
                layer_weights.down_bias,
                route_outputs,
                hidden_size,
                local_intermediate_size,
                1,
                True,
            ),
        ):
            mxfp4_expert_matmul_kernel[(expert_block_count, triton.cdiv(width, 64))](
                inputs,
                blocks,
                scales,
                bias,
                destination,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                N=typing.cast(language.constexpr, width),
                K=typing.cast(language.constexpr, reduction),
                EXPERT_COUNT=typing.cast(language.constexpr, expert_count),
                ROUTE_COUNT=typing.cast(language.constexpr, route_count),
                INPUT_TOPK=typing.cast(language.constexpr, input_topk),
                MULTIPLY_ROUTE=typing.cast(language.constexpr, multiply),
                BLOCK_M=typing.cast(language.constexpr, block_size_m),
                BLOCK_N=typing.cast(language.constexpr, 64),
                BLOCK_K=typing.cast(language.constexpr, 64),
                num_warps=4,
                num_stages=3,
            )
            if not multiply:
                compute_clamped_swiglu(
                    gate_up=gate_up,
                    output=activated,
                    alpha=typing.cast(float, activation_alpha),
                    clamp_limit=typing.cast(float, activation_clamp_limit),
                )
        sgl_kernel.moe_sum_reduce(route_outputs, output, routed_scaling_factor)
        return output

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel

    # Pinned SGLang exposes this launcher positionally: W13 consumes hidden
    # states and semantic routes, then writes the caller-owned Gate/Up tensor.
    invoke_fused_moe_kernel(
        hidden_states,
        layer_weights.expert_gate_up_weight,
        None,
        gate_up,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        effective_topk,
        w13,
        output_dtype,
        False,
        False,
        False,
        False,
        False,
        filter_expert=True,
    )
    returned_activation = sgl_kernel.silu_and_mul(gate_up, out=activated)
    if returned_activation is not activated:
        raise RuntimeError("sgl_kernel.silu_and_mul did not preserve caller-owned output")
    # W2 reuses the same aligned route metadata and writes one output per route;
    # the following combine is the only row-level reduction.
    invoke_fused_moe_kernel(
        activated,
        layer_weights.expert_down_weight,
        None,
        route_outputs,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        w2,
        output_dtype,
        False,
        False,
        False,
        False,
        False,
        filter_expert=True,
    )
    sgl_kernel.moe_sum_reduce(route_outputs, output, routed_scaling_factor)
    return output


def compute_dense_partial(
    *,
    hidden_states: torch.Tensor,
    layer_weights: weights.DenseFfnWeights,
    workspace: torch.Tensor,
    output: torch.Tensor,
    activation: ActivationKind,
) -> torch.Tensor:
    """Compute one fixed-Capacity rank-local gated Dense Partial.

    Args:
        hidden_states: Caller-owned contiguous payload input shaped ``[C, H]``.
        layer_weights: Canonical local true-TP projection weights.
        workspace: Caller-owned contiguous CUDA byte tensor with the exact
        result of :func:`execution.dense_workspace_bytes`.
        output: Caller-owned contiguous payload destination shaped ``[C, H]``.
        activation: Startup-selected gated activation semantic.

    Returns:
        The exact ``output`` object after all Capacity rows are evaluated.

    Raises:
        ValueError: If activation, shapes, dtype, device, contiguity, or
            alignment do not match the fixed Dense implementation.
        ImportError: If the direct ``sgl_kernel`` dependency cannot load.

    Side Effects:
        Writes ``workspace`` and ``output`` on the current CUDA stream. It
        allocates no Torch output or persistent auxiliary tensor.
    """

    if activation is not ActivationKind.SILU:
        raise ValueError(f"unsupported Dense activation {activation!r}")
    payload_dtype = hidden_states.dtype
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Dense execution requires BF16 or FP16 payloads")
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(output, name="Dense Partial output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(workspace, name="Dense workspace", dtype=torch.uint8, dimensions=1)

    row_capacity, hidden_size = hidden_states.shape
    local_intermediate_twice, weight_hidden_size = layer_weights.gate_up_weight.shape
    local_intermediate_size = local_intermediate_twice // 2
    if (
        weight_hidden_size != hidden_size
        or layer_weights.gate_up_weight.dtype is not payload_dtype
        or output.shape != hidden_states.shape
    ):
        raise ValueError("Dense input, output, and weight dimensions disagree")
    expected_workspace_bytes = execution.dense_workspace_bytes(
        payload_dtype=payload_dtype,
        row_capacity=row_capacity,
        local_intermediate_size=local_intermediate_size,
    )
    if workspace.numel() != expected_workspace_bytes:
        raise ValueError(f"Dense workspace must contain exactly {expected_workspace_bytes} bytes")
    device = hidden_states.device
    if output.device != device or workspace.device != device or layer_weights.gate_up_weight.device != device:
        raise ValueError("Dense input, output, workspace, and weights must share one device")
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("Dense Partial output", output),
        ("Dense workspace", workspace),
        ("gate/up weight", layer_weights.gate_up_weight),
        ("down weight", layer_weights.down_weight),
    ):
        if tensor.data_ptr() % execution.OPERATOR_ALIGNMENT_BYTES != 0:
            raise ValueError(f"{name} must be 16-byte aligned")
    if local_intermediate_twice * payload_dtype.itemsize % execution.OPERATOR_ALIGNMENT_BYTES != 0:
        raise ValueError("Dense gate/up rows must be 16-byte aligned")

    workspace_values = workspace.view(payload_dtype)
    gate_up_elements = row_capacity * local_intermediate_twice
    gate_up = workspace_values[:gate_up_elements].view(row_capacity, local_intermediate_twice)
    activated_up = workspace_values[gate_up_elements:].view(row_capacity, local_intermediate_size)

    import sgl_kernel

    torch.mm(hidden_states, layer_weights.gate_up_weight.t(), out=gate_up)
    returned_output = sgl_kernel.silu_and_mul(gate_up, out=activated_up)
    if returned_output is not activated_up:
        raise RuntimeError("sgl_kernel.silu_and_mul did not preserve caller-owned output")
    torch.mm(activated_up, layer_weights.down_weight.t(), out=output)
    return output
