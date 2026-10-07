"""Parameter-free pinned SGLang GPT-OSS FFN replacement."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.gpt_oss import GptOssForCausalLM, GptOssSparseMoeBlock
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_parallel
from transformers import GptOssConfig

from xpool.integrations.sglang.adapter import SglangShimAdapter, model_runner_architectures
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

GPT_OSS_FFN_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp(?:\.|$)")


def validate_gpt_oss_profile(config: GptOssConfig) -> None:
    """Admit only the exact 20B activation and geometry on the attention side."""

    if (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_local_experts,
        config.num_experts_per_tok,
    ) != (2880, 2880, 24, 32, 4):
        raise ShimUnavailableError("xpool GPT-OSS shim supports only the verified 20B geometry")
    if config.hidden_act != "silu" or getattr(config, "hidden_act_alpha", 1.702) != 1.702 or config.swiglu_limit != 7.0:
        raise ShimUnavailableError("xpool GPT-OSS shim requires alpha=1.702 and swiglu_limit=7")


class XpoolGptOssSparseMoeBlock(FfnShimModule, GptOssSparseMoeBlock):
    """Replace every sparse decoder block without constructing Expert parameters."""

    def __init__(
        self,
        layer_id: int,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Match the pinned constructor while initializing only shim state."""

        validate_gpt_oss_profile(config)
        if quant_config is None or quant_config.get_name() != "mxfp4":
            raise ShimUnavailableError("xpool GPT-OSS shim requires official MXFP4 quantization")
        FfnShimModule.__init__(self, layer_id=layer_id, hidden_size=config.hidden_size, layer_kind=LayerKind.MOE)

    def get_moe_weights(self) -> list[torch.Tensor]:
        """Satisfy original GPT-OSS routed-weight discovery with an empty tensor set."""

        return []


def filter_gpt_oss_ffn_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Remove canonical MXFP4 Expert and Router keys before the original loader.

    SGLang's MXFP4 loader consumes the iterator to separate Expert tensors.
    With no Expert tensors it performs no Expert parameter lookup, then calls
    its unchanged normal loader for attention, sinks, norms and embeddings.
    Legacy block-schema FFNs are rejected because only the official HF layout
    is admitted; legacy norm weights must not be mistaken for FFN resources.
    """

    for name, tensor in weights:
        if name.startswith("block."):
            raise ShimUnavailableError("xpool GPT-OSS requires the official model.layers checkpoint layout")
        if GPT_OSS_FFN_PATTERN.match(name) is None:
            yield name, tensor


def around_load_weights(
    original_fn: Callable[..., object],
    model: GptOssForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
    weight_name_mapping: dict[str, str] | None = None,
) -> object:
    """Preserve GPT-OSS's special MXFP4/normal loader control flow."""

    if is_nextn or weight_name_mapping:
        raise ShimUnavailableError("xpool GPT-OSS shim does not support next-token modules or renamed checkpoints")
    return original_fn(
        model, filter_gpt_oss_ffn_weights(weights), is_nextn=is_nextn, weight_name_mapping=weight_name_mapping
    )


class GptOssShimAdapter(SglangShimAdapter):
    """Own GPT-OSS-specific hooks and conservative attention topology admission."""

    name = "gpt_oss"
    supports_dp_attention = False

    def hooks(self) -> tuple[SglangHook, ...]:
        return (
            SglangHook(
                target="sglang.srt.models.gpt_oss.GptOssSparseMoeBlock",
                handler=XpoolGptOssSparseMoeBlock,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.gpt_oss.GptOssForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        return "GptOssForCausalLM" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Require the existing full-storage KV boundary for sliding attention.

        Sliding-window attention math remains SGLang-owned. Its separate
        hybrid SWA allocator is outside CrossPool's elastic pool contract.
        """

        if model_runner.model_config.dtype is not torch.bfloat16:
            raise ShimUnavailableError("xpool GPT-OSS requires --dtype bfloat16")
        if get_parallel().attn_tp_size != 1:
            raise ShimUnavailableError("xpool GPT-OSS first profile requires attention TP 1")
        if model_runner.model_config.is_hybrid_swa:
            raise ShimUnavailableError(
                "xpool GPT-OSS requires --disable-hybrid-swa-memory for the elastic full-storage KV pool"
            )

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require all 24 decoder FFNs to be parameter-free FULL-boundary shims."""

        model = model_runner.model
        if not isinstance(model, GptOssForCausalLM) or not isinstance(model.config, GptOssConfig):
            raise RuntimeError("xpool GPT-OSS runner did not load GptOssForCausalLM/GptOssConfig")
        validate_gpt_oss_profile(model.config)
        shims = self.require_ffn_shims(
            model, expected_layer_kinds=(LayerKind.MOE,) * 24, allowed_shim_types=(XpoolGptOssSparseMoeBlock,)
        )
        if any(tuple(shim.parameters()) for shim in shims):
            raise RuntimeError("xpool GPT-OSS attention-side FFNs retain parameters")
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)
