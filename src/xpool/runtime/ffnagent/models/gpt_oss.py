"""Strict official GPT-OSS-20B MXFP4 FFN semantics."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, operators, weights
from xpool.utils import align_up


class GptOssAdapter(architecture.MoeFfnModelAdapter):
    """Compile only the inspected GPT-OSS-20B checkpoint profile."""

    architecture_name = "GptOssForCausalLM"

    @staticmethod
    def router_weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype:
        """Keep the reference's BF16 Router weight and additive bias."""

        if payload_dtype is not torch.bfloat16:
            raise ValueError("GPT-OSS MXFP4 requires BF16 payloads")
        return torch.bfloat16

    @staticmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
        hidden_size: int,
        routed_expert_count: int,
        routed_topk: int,
    ) -> int:
        """Return caller-owned BF16 logits for the admitted Router geometry."""

        GptOssAdapter.router_weight_dtype(payload_dtype=payload_dtype)
        if payload_row_capacity <= 0 or (hidden_size, routed_expert_count, routed_topk) != (2880, 32, 4):
            raise ValueError("GPT-OSS Router geometry is outside the admitted profile")
        logits_bytes = payload_row_capacity * routed_expert_count * payload_dtype.itemsize
        carrier_bytes = align_up(payload_row_capacity * routed_topk * 2, 16)
        return logits_bytes + 2 * carrier_bytes + align_up(payload_row_capacity, 32) * 4

    @staticmethod
    def compute_routed_topk(
        *,
        hidden_states: torch.Tensor,
        router_weights: weights.MoeRouterWeights,
        workspace: torch.Tensor,
        routed_ids: torch.Tensor,
        routed_weights: torch.Tensor,
        renormalize: bool,
    ) -> None:
        """Project with ordinary bias, then normalize selected softmax routes."""

        rows, hidden_size = hidden_states.shape
        expert_count = router_weights.weight.shape[0]
        required = GptOssAdapter.router_workspace_bytes(
            payload_dtype=hidden_states.dtype,
            payload_row_capacity=rows,
            hidden_size=hidden_size,
            routed_expert_count=expert_count,
            routed_topk=routed_ids.shape[1],
        )
        if workspace.dtype is not torch.uint8 or workspace.numel() != required or not workspace.is_contiguous():
            raise ValueError("GPT-OSS Router workspace disagrees with the fixed geometry")
        if router_weights.projection_bias is None or router_weights.correction_bias is not None or not renormalize:
            raise ValueError("GPT-OSS requires additive Router bias and normalized routing")
        if (
            router_weights.weight.shape != (expert_count, hidden_size)
            or router_weights.weight.dtype is not torch.bfloat16
        ):
            raise ValueError("GPT-OSS Router weight geometry disagrees")
        weights.validate_tensor(hidden_states, name="GPT-OSS hidden states", dtype=torch.bfloat16, dimensions=2)
        weights.validate_tensor(routed_ids, name="GPT-OSS routes", dtype=torch.int32, dimensions=2)
        weights.validate_tensor(routed_weights, name="GPT-OSS route weights", dtype=torch.float32, dimensions=2)
        if routed_ids.shape != (rows, 4) or routed_weights.shape != routed_ids.shape:
            raise ValueError("GPT-OSS routing destinations disagree")
        tensors = (workspace, router_weights.weight, router_weights.projection_bias, routed_ids, routed_weights)
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("GPT-OSS Router tensors must share one device")
        logits_bytes = rows * expert_count * torch.bfloat16.itemsize
        logits = workspace[:logits_bytes].view(torch.bfloat16).view(rows, expert_count)
        operators.compute_biased_router_logits(
            hidden_states=hidden_states, router_weights=router_weights, logits=logits
        )
        operators.compute_bf16_selected_softmax_topk(
            logits=logits,
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            workspace=workspace[logits_bytes:],
        )

    @classmethod
    def compile(cls, *, model_id: ModelId, model_config: architecture.FfnSourceConfig) -> ffn.FfnModelSpec:
        """Reject unverified profiles instead of accepting a family name alone."""

        if model_config.architecture_name != cls.architecture_name:
            raise ValueError("GPT-OSS requires GptOssForCausalLM")
        if model_config.get("model_type", str) != "gpt_oss" or model_config.get("hidden_act", str) != "silu":
            raise ValueError("GPT-OSS model type or activation disagrees with the admitted profile")
        expected = {
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
        }
        for name, value in expected.items():
            if model_config.get(name, int, ge=1) != value:
                raise ValueError(f"GPT-OSS {name} must equal {value}")
        alias = model_config.optional("experts_per_token", int)
        if alias is not None and alias != 4:
            raise ValueError("GPT-OSS experts_per_token conflicts with num_experts_per_tok")
        alpha = model_config.optional("hidden_act_alpha", float)
        if alpha is not None and alpha != 1.702:
            raise ValueError("GPT-OSS hidden_act_alpha must equal 1.702")
        if model_config.get("swiglu_limit", float) != 7.0:
            raise ValueError("GPT-OSS swiglu_limit must equal 7.0")
        for name in ("dtype", "torch_dtype"):
            dtype = model_config.optional(name, str)
            if dtype is not None and dtype != "bfloat16":
                raise ValueError(f"GPT-OSS {name} must equal bfloat16 when declared")
        quantization = model_config.get("quantization_config", dict)
        if quantization.get("quant_method") != "mxfp4":
            raise ValueError("GPT-OSS requires the official MXFP4 checkpoint")
        excluded = quantization.get("modules_to_not_convert")
        if (
            not isinstance(excluded, list)
            or not all(isinstance(name, str) for name in excluded)
            or len(excluded) != 4
            or set(excluded)
            != {
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            }
        ):
            raise ValueError("GPT-OSS quantization exclusions disagree with the official profile")
        layers = []
        for layer_id in range(24):
            prefix = f"model.layers.{layer_id}.mlp"
            layers.append(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    layer_id=layer_id,
                    expert_intermediate_size=2880,
                    shared_expert_count=0,
                    routed_topk=4,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    expert_weight_kind=ffn.ExpertWeightKind.MXFP4,
                    checkpoint=ffn.MoeFfnCheckpointKeys(
                        router_weight_key=f"{prefix}.router.weight",
                        router_projection_bias_key=f"{prefix}.router.bias",
                        router_correction_bias_key=None,
                        shared_expert=None,
                        mxfp4_experts=ffn.Mxfp4ExpertCheckpointKeys(
                            expert_count=32,
                            gate_up_blocks_key=f"{prefix}.experts.gate_up_proj_blocks",
                            down_blocks_key=f"{prefix}.experts.down_proj_blocks",
                            gate_up_scales_key=f"{prefix}.experts.gate_up_proj_scales",
                            down_scales_key=f"{prefix}.experts.down_proj_scales",
                            gate_up_bias_key=f"{prefix}.experts.gate_up_proj_bias",
                            down_bias_key=f"{prefix}.experts.down_proj_bias",
                        ),
                    ),
                )
            )
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=2880,
            activation=ffn.ActivationKind.CLAMPED_SWIGLU,
            activation_alpha=1.702,
            activation_clamp_limit=7.0,
            layers=tuple(layers),
        )
