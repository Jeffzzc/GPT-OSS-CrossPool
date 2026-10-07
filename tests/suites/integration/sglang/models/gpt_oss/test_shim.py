"""Pinned GPT-OSS parameter-free constructor and original MXFP4 loader control flow."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.gpt_oss import GptOssForCausalLM, GptOssSparseMoeBlock
from torch import nn
from transformers import GptOssConfig

from xpool.integrations.sglang.models import gpt_oss
from xpool.integrations.sglang.models.gpt_oss import (
    GptOssShimAdapter,
    XpoolGptOssSparseMoeBlock,
    around_load_weights,
    filter_gpt_oss_ffn_weights,
)
from xpool.integrations.sglang.shim import ShimUnavailableError
from xtest.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


def config() -> GptOssConfig:
    """Build the exact admitted geometry using the pinned config class."""

    return GptOssConfig(
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=24,
        num_local_experts=32,
        num_experts_per_tok=4,
        swiglu_limit=7.0,
    )


def test_constructor_matches_pinned_surface_and_retains_no_experts() -> None:
    original = inspect.signature(GptOssSparseMoeBlock.__init__)
    replacement = inspect.signature(XpoolGptOssSparseMoeBlock.__init__)
    assert tuple(original.parameters) == tuple(replacement.parameters)
    for layer_id in range(24):
        shim = XpoolGptOssSparseMoeBlock(
            layer_id=layer_id, config=config(), quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
        )
        assert isinstance(shim, GptOssSparseMoeBlock)
        assert tuple(shim.parameters()) == () and shim.get_moe_weights() == []
        assert not hasattr(shim, "experts")


def test_original_special_loader_keeps_non_ffn_weights() -> None:
    """Call the unmodified pinned load_weights method rather than a stand-in loader."""

    model = GptOssForCausalLM.__new__(GptOssForCausalLM)
    nn.Module.__init__(model)
    model.config = config()
    model.quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
    model.model = nn.Module()
    model.lm_head = nn.Linear(4, 4, bias=False)
    expected = torch.arange(16, dtype=model.lm_head.weight.dtype).view(4, 4)
    ffns = (
        (f"model.layers.{layer}.mlp.{suffix}", torch.ones(1))
        for layer in range(24)
        for suffix in (
            "router.weight",
            "router.bias",
            "experts.gate_up_proj_blocks",
            "experts.gate_up_proj_scales",
            "experts.gate_up_proj_bias",
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
            "experts.down_proj_bias",
        )
    )
    around_load_weights(GptOssForCausalLM.load_weights, model, iter((*ffns, ("lm_head.weight", expected))))
    torch.testing.assert_close(model.lm_head.weight, expected, rtol=0, atol=0)
    assert tuple(name for name, _ in model.named_parameters()) == ("lm_head.weight",)


def test_filter_preserves_attention_bias_sinks_and_norms() -> None:
    names = (
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.input_layernorm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    )
    tensors = [(name, torch.ones(1)) for name in names]
    tensors.append(("model.layers.0.mlp.router.bias", torch.ones(1)))
    assert tuple(name for name, _ in filter_gpt_oss_ffn_weights(tensors)) == names
    with pytest.raises(ShimUnavailableError, match="official"):
        tuple(filter_gpt_oss_ffn_weights((("block.0.mlp.weight", torch.ones(1)),)))


@pytest.mark.parametrize(
    ("dtype", "hybrid_swa", "attention_tp", "error"),
    (
        (torch.bfloat16, False, 1, None),
        (torch.float16, False, 1, "bfloat16"),
        (torch.bfloat16, True, 1, "disable-hybrid-swa-memory"),
        (torch.bfloat16, False, 2, "attention TP 1"),
    ),
)
def test_first_profile_fails_closed_before_loading(
    dtype: torch.dtype, hybrid_swa: bool, attention_tp: int, error: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only declared admission metadata is needed; constructing the real runner
    # would initialize distributed groups and CUDA before the boundary under test.
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_config = ModelConfig.__new__(ModelConfig)
    runner.model_config.dtype = dtype
    runner.model_config.is_hybrid_swa = hybrid_swa
    monkeypatch.setattr(gpt_oss, "get_parallel", lambda: SimpleNamespace(attn_tp_size=attention_tp))
    if error is None:
        GptOssShimAdapter().validate_before_load(runner)
    else:
        with pytest.raises(ShimUnavailableError, match=error):
            GptOssShimAdapter().validate_before_load(runner)


def test_after_load_requires_every_decoder_ffn_and_full_boundaries() -> None:
    hf_config = config()
    quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
    layers = [
        FakeDecoderLayer(
            XpoolGptOssSparseMoeBlock(layer_id=layer, config=hf_config, quant_config=quant_config),
            allow_reduce_scatter=True,
        )
        for layer in range(24)
    ]
    runner = runner_with_architecture("GptOssForCausalLM")
    runner.model = loaded_model(GptOssForCausalLM, hf_config, layers)
    adapter = GptOssShimAdapter()
    assert adapter.matches(runner.as_model_runner()) and not adapter.supports_dp_attention
    adapter.validate_after_load(runner.as_model_runner())
    layers[12].mlp = nn.Linear(2880, 1)
    with pytest.raises(RuntimeError, match="layer ids"):
        adapter.validate_after_load(runner.as_model_runner())
