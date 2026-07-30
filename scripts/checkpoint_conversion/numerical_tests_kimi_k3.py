# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity for the reduced TorchTitan and released HF Kimi K3 models.

The released checkpoint contains MXFP4 routed-expert weights, so this test
constructs the same reduced, unquantized topology on both sides. TorchTitan
initializes the weights once, ``KimiK3StateDictAdapter`` converts them to the
released HuggingFace schema, and the HuggingFace model loads them strictly.

The comparison covers:

- text-only decoder layers, KDA/MLA outputs, router expert IDs, and logits;
- vision transformer blocks and projected vision features;
- end-to-end image+text decoder layers and logits.

The released HuggingFace implementation imports FLA for KDA. By default, run
this script in an environment that satisfies the released model requirements
and has a CUDA GPU. The script requests the released eager MLA and
vision-attention paths after construction so the comparison isolates model math
from FlashAttention kernels.

For a CPU-only comparison, ``--hf_kda_backend reference`` installs the minimum
released FLA API with pure PyTorch operators.

Example:

    CUDA_VISIBLE_DEVICES=0 python -m \
        scripts.checkpoint_conversion.numerical_tests_kimi_k3 \
        --hf_repo_path ~/hf_assets/moonshotai/Kimi-K3

    python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 \
        --hf_repo_path ~/hf_assets/moonshotai/Kimi-K3 \
        --device cpu \
        --hf_kda_backend reference
"""

import argparse
import importlib
import sys
import types
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from transformers import AutoConfig, AutoModelForCausalLM


_IMAGE_TOKEN_ID = 7


class _ReferenceShortConvolution(nn.Module):
    """Pure PyTorch compatibility layer for the released HF KDA module."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str = "silu",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.empty(hidden_size)) if bias else None
        self.kernel_size = kernel_size
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del cache, cu_seqlens
        output = F.conv1d(
            F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0)),
            self.weight,
            self.bias,
            groups=self.weight.shape[0],
        ).transpose(1, 2)
        if self.activation == "silu":
            output = F.silu(output)
        final_state = x[:, -self.kernel_size + 1 :] if output_final_state else None
        return output, final_state


class _ReferenceRMSNormGated(nn.Module):
    """Pure PyTorch equivalent of FLA's fused gated RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "sigmoid",
    ):
        super().__init__()
        if activation != "sigmoid":
            raise ValueError(
                "The Kimi K3 reference RMSNorm only supports a sigmoid gate."
            )
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float * self.weight.float() * torch.sigmoid(gate.float())).to(
            input_dtype
        )


def _reference_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    lower_bound: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure PyTorch equivalent of the FLA KDA inference API."""
    del kwargs
    input_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    log_decay = lower_bound * torch.sigmoid(
        torch.exp(A_log.float()).view(1, 1, -1, 1)
        * (g.float() + dt_bias.float().view(1, 1, *g.shape[-2:]))
    )
    beta = torch.sigmoid(beta.float())

    batch_size, seq_len, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]
    state = (
        torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for token_idx in range(seq_len):
        state = state * torch.exp(log_decay[:, token_idx]).unsqueeze(-1)
        old_value = torch.matmul(
            k[:, token_idx].unsqueeze(-2),
            state,
        ).squeeze(-2)
        delta = (v[:, token_idx] - old_value) * beta[:, token_idx].unsqueeze(-1)
        state = state + k[:, token_idx].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            torch.matmul(q[:, token_idx].unsqueeze(-2), state).squeeze(-2)
            * (head_dim**-0.5)
        )
    output = torch.stack(outputs, dim=1).to(input_dtype)
    return output, state if output_final_state else None


def _install_reference_fla() -> None:
    """Expose the minimum FLA API used by the released HF model."""

    # The released vision reference decorates an interpolation helper with
    # torch.compile. Keep the compatibility backend fully eager so it remains
    # usable for the CPU-only comparison.
    def eager_compile(model=None, *args, **kwargs):
        del args, kwargs
        return model if model is not None else lambda wrapped: wrapped

    torch.__dict__["compile"] = eager_compile

    fla = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_modules.__dict__.update(
        {
            "ShortConvolution": _ReferenceShortConvolution,
            "FusedRMSNormGated": _ReferenceRMSNormGated,
        }
    )
    fla_ops = types.ModuleType("fla.ops")
    fla_kda = types.ModuleType("fla.ops.kda")
    fla_kda.__dict__.update(
        {
            "chunk_kda": _reference_kda,
            "fused_recurrent_kda": _reference_kda,
        }
    )
    fla_ops_utils = types.ModuleType("fla.ops.utils")
    fla_ops_index = types.ModuleType("fla.ops.utils.index")
    fla_ops_index.__dict__.update(
        {
            "prepare_cu_seqlens_from_mask": lambda _mask: None,
            "prepare_lens_from_mask": lambda _mask: None,
        }
    )
    fla_utils = types.ModuleType("fla.utils")
    fla_utils.__dict__["tensor_cache"] = lambda fn: fn

    for module in (
        fla,
        fla_modules,
        fla_ops,
        fla_kda,
        fla_ops_utils,
        fla_ops_index,
        fla_utils,
    ):
        sys.modules[module.__name__] = module


def _build_reduced_hf_config(hf_repo_path: str, tt_config):
    """Build the released HF config with TorchTitan's reduced dimensions."""
    released_config = AutoConfig.from_pretrained(
        hf_repo_path,
        trust_remote_code=True,
    )
    text_config_cls = type(released_config.text_config)
    vision_config_cls = type(released_config.vision_config)
    config_cls = type(released_config)

    mla_config = next(
        layer.attention for layer in tt_config.layers if layer.attention is not None
    )
    kda_config = next(
        layer.delta_attention
        for layer in tt_config.layers
        if layer.delta_attention is not None
    )
    dense_config = next(
        layer.feed_forward
        for layer in tt_config.layers
        if layer.feed_forward is not None
    )
    moe_config = next(layer.moe for layer in tt_config.layers if layer.moe is not None)
    vision_config = tt_config.vision_encoder
    if vision_config is None:
        raise ValueError("The Kimi K3 debug config must include a vision encoder.")

    full_attention_layers = [
        layer_idx + 1
        for layer_idx, layer in enumerate(tt_config.layers)
        if layer.attention is not None
    ]
    kda_layers = [
        layer_idx + 1
        for layer_idx, layer in enumerate(tt_config.layers)
        if layer.delta_attention is not None
    ]
    first_moe_layer = next(
        layer_idx
        for layer_idx, layer in enumerate(tt_config.layers)
        if layer.moe is not None
    )

    text_config = text_config_cls(
        vocab_size=tt_config.vocab_size,
        hidden_size=tt_config.dim,
        intermediate_size=dense_config.w1.out_features,
        num_hidden_layers=len(tt_config.layers),
        num_attention_heads=mla_config.num_heads,
        num_key_value_heads=mla_config.num_heads,
        hidden_act="situ",
        rms_norm_eps=tt_config.norm.eps,
        use_cache=False,
        moe_intermediate_size=moe_config.routed_experts[0].w1.out_features,
        num_experts=moe_config.num_experts,
        num_experts_per_token=moe_config.router.top_k,
        num_shared_experts=(
            moe_config.shared_experts.w1.out_features
            // moe_config.routed_experts[0].w1.out_features
        ),
        first_k_dense_replace=first_moe_layer,
        moe_layer_freq=1,
        moe_renormalize=moe_config.router.route_norm,
        routed_scaling_factor=moe_config.router.route_scale,
        num_expert_group=1,
        topk_group=1,
        q_lora_rank=mla_config.q_lora_rank,
        kv_lora_rank=mla_config.kv_lora_rank,
        qk_nope_head_dim=mla_config.qk_nope_head_dim,
        qk_rope_head_dim=mla_config.qk_rope_head_dim,
        v_head_dim=mla_config.v_head_dim,
        mla_use_nope=True,
        mla_use_output_gate=True,
        linear_attn_config={
            "kda_layers": kda_layers,
            "full_attn_layers": full_attention_layers,
            "short_conv_kernel_size": kda_config.conv_kernel_size,
            "head_dim": kda_config.head_dim,
            "num_heads": kda_config.num_heads,
            "use_full_rank_gate": True,
            "gate_lower_bound": kda_config.kernel.lower_bound,
        },
        attn_res_block_size=tt_config.layers[0].attn_res_block_size,
        latent_moe_use_norm=True,
        activation_situ_beta=dense_config.activation.beta,
        activation_situ_linear_beta=dense_config.activation.linear_beta,
        routed_expert_hidden_size=moe_config.routed_down.out_features,
    )
    text_config._attn_implementation = "eager"

    vision_attention = vision_config.block.attn
    vision_mlp = vision_config.block.mlp
    vision_config_hf = vision_config_cls(
        patch_size=vision_config.patch_size,
        init_pos_emb_height=vision_config.init_pos_emb_height,
        init_pos_emb_width=vision_config.init_pos_emb_width,
        init_pos_emb_time=vision_config.max_num_frames,
        vt_num_attention_heads=vision_attention.num_heads,
        vt_num_hidden_layers=vision_config.num_layers,
        vt_hidden_size=vision_config.dim,
        vt_intermediate_size=vision_mlp.fc1.out_features,
        merge_kernel_size=tuple(vision_config.merge_kernel_size),
        mm_projector_type="patchmergerv2",
        qkv_hidden_size=vision_attention.qkv_dim,
        text_hidden_size=tt_config.dim,
        norm_type="rmsnorm",
        attn_bias=False,
        patch_embed_proj_bias=False,
        linear_bias=False,
        activation_func="gelu_pytorch_tanh",
        pos_emb_interpolation_mode=vision_config.interpolation_mode,
    )
    vision_config_hf._attn_implementation = "eager"

    config = config_cls(
        text_config=text_config,
        vision_config=vision_config_hf,
        media_placeholder_token_id=_IMAGE_TOKEN_ID,
        pad_token_id=0,
        auto_map=released_config.auto_map,
    )
    config._name_or_path = hf_repo_path
    return config


def _first_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Expected a tensor output, got {type(output).__name__}.")


def _capture_tensor(
    destination: dict[str, torch.Tensor],
    name: str,
    *,
    flatten: bool = False,
) -> Callable:
    def hook(_module, _inputs, output) -> None:
        value = _first_tensor(output).detach().float().cpu()
        if flatten:
            value = value.reshape(-1, value.shape[-1])
        destination[name] = value

    return hook


def _capture_router_ids(
    destination: dict[str, torch.Tensor],
    name: str,
) -> Callable:
    def hook(_module, _inputs, output) -> None:
        expert_ids = output[0]
        destination[name] = expert_ids.detach().reshape(-1, expert_ids.shape[-1]).cpu()

    return hook


def _register_text_hooks(
    tt_model,
    hf_model,
    tt_outputs: dict[str, torch.Tensor],
    hf_outputs: dict[str, torch.Tensor],
) -> None:
    for layer_idx, tt_layer in enumerate(tt_model.layers.values()):
        hf_layer = hf_model.language_model.model.layers[layer_idx]
        layer_name = f"decoder.layer.{layer_idx}"
        tt_layer.register_forward_hook(_capture_tensor(tt_outputs, layer_name))
        hf_layer.register_forward_hook(_capture_tensor(hf_outputs, layer_name))

        attention_name = f"decoder.attention.{layer_idx}"
        tt_attention = (
            tt_layer.attention
            if tt_layer.attention is not None
            else tt_layer.delta_attention
        )
        tt_attention.register_forward_hook(_capture_tensor(tt_outputs, attention_name))
        hf_layer.self_attn.register_forward_hook(
            _capture_tensor(hf_outputs, attention_name)
        )

        if tt_layer.moe is not None:
            router_name = f"decoder.router_ids.{layer_idx}"
            tt_layer.moe.router.register_forward_hook(
                _capture_router_ids(tt_outputs, router_name)
            )
            hf_layer.block_sparse_moe.gate.register_forward_hook(
                _capture_router_ids(hf_outputs, router_name)
            )


def _register_vision_hooks(
    tt_model,
    hf_model,
    tt_outputs: dict[str, torch.Tensor],
    hf_outputs: dict[str, torch.Tensor],
) -> None:
    for layer_idx, tt_layer in enumerate(tt_model.vision_encoder.layers.values()):
        layer_name = f"vision.layer.{layer_idx}"
        tt_layer.register_forward_hook(
            _capture_tensor(tt_outputs, layer_name, flatten=True)
        )
        hf_model.vision_tower.encoder.blocks[layer_idx].register_forward_hook(
            _capture_tensor(hf_outputs, layer_name, flatten=True)
        )


def _compare_tensor(
    name: str,
    tt_value: torch.Tensor,
    hf_value: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    if tt_value.shape != hf_value.shape:
        raise AssertionError(
            f"{name}: TorchTitan shape {tuple(tt_value.shape)} does not match "
            f"HF shape {tuple(hf_value.shape)}."
        )
    difference = (tt_value - hf_value).abs()
    cosine = F.cosine_similarity(
        tt_value.reshape(-1),
        hf_value.reshape(-1),
        dim=0,
    ).item()
    print(
        f"{name:32s} max={difference.max().item():.4e} "
        f"mean={difference.mean().item():.4e} cos={cosine:.8f}"
    )
    torch.testing.assert_close(
        tt_value,
        hf_value,
        atol=atol,
        rtol=rtol,
        msg=lambda message: f"{name} failed numerical parity:\n{message}",
    )


def _compare_captured(
    tt_outputs: dict[str, torch.Tensor],
    hf_outputs: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    if tt_outputs.keys() != hf_outputs.keys():
        raise AssertionError(
            "Captured output names differ: "
            f"TT-only={sorted(tt_outputs.keys() - hf_outputs.keys())}, "
            f"HF-only={sorted(hf_outputs.keys() - tt_outputs.keys())}."
        )
    for name in tt_outputs:
        if ".router_ids." in name:
            if not torch.equal(tt_outputs[name], hf_outputs[name]):
                mismatch = (tt_outputs[name] != hf_outputs[name]).sum().item()
                raise AssertionError(f"{name}: {mismatch} expert IDs differ.")
            print(f"{name:32s} exact expert IDs")
        else:
            _compare_tensor(
                name,
                tt_outputs[name],
                hf_outputs[name],
                atol=atol,
                rtol=rtol,
            )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tt_device",
        help="TorchTitan device. Defaults to --device.",
    )
    parser.add_argument(
        "--hf_device",
        help="HuggingFace device. Defaults to --device.",
    )
    parser.add_argument(
        "--device_module",
        help="Optional module that registers an out-of-tree PyTorch device.",
    )
    parser.add_argument(
        "--hf_kda_backend",
        default="fla",
        choices=("fla", "reference"),
        help="Use FLA or the script's pure PyTorch KDA compatibility backend.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
    )
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device_module is not None:
        importlib.import_module(args.device_module)
    tt_device = torch.device(args.tt_device or args.device)
    hf_device = torch.device(args.hf_device or args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if args.hf_kda_backend == "reference":
        _install_reference_fla()
    if "cuda" in {tt_device.type, hf_device.type}:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    tt_config = cast(
        KimiK3Model.Config,
        model_registry("debugmodel").model,
    )
    tt_model = tt_config.build()
    tt_model.init_states()

    hf_config = _build_reduced_hf_config(args.hf_repo_path, tt_config)
    hf_model = AutoModelForCausalLM.from_config(
        hf_config,
        trust_remote_code=True,
    )
    hf_state_dict = KimiK3StateDictAdapter(
        tt_config,
        hf_assets_path=None,
    ).to_hf(tt_model.state_dict())
    hf_model.load_state_dict(hf_state_dict, strict=True)

    # KimiLinearModel selects FlashAttention during construction. Restoring
    # eager here exercises the released eager attention function.
    hf_model.language_model.model.config._attn_implementation = "eager"
    hf_model.config.text_config._attn_implementation = "eager"

    tt_model = tt_model.to(device=tt_device, dtype=dtype).eval()
    hf_model = hf_model.to(device=hf_device, dtype=dtype).eval()

    tt_outputs: dict[str, torch.Tensor] = {}
    hf_outputs: dict[str, torch.Tensor] = {}
    _register_text_hooks(tt_model, hf_model, tt_outputs, hf_outputs)
    _register_vision_hooks(tt_model, hf_model, tt_outputs, hf_outputs)

    print("\nText-only parity")
    tokens_BL = torch.tensor(
        [[11, 23, 17, 31, 5, 19, 29, 3]],
        dtype=torch.long,
    )
    tt_tokens_BL = tokens_BL.to(tt_device)
    hf_tokens_BL = tokens_BL.to(hf_device)
    attention_mask_BL = torch.ones_like(hf_tokens_BL)
    tt_logits_BLV = tt_model(tt_tokens_BL).float().cpu()
    hf_logits_BLV = (
        hf_model.language_model(
            input_ids=hf_tokens_BL,
            attention_mask=attention_mask_BL,
            use_cache=False,
        )
        .logits.float()
        .cpu()
    )
    _compare_captured(
        tt_outputs,
        hf_outputs,
        atol=args.atol,
        rtol=args.rtol,
    )
    _compare_tensor(
        "text.logits",
        tt_logits_BLV,
        hf_logits_BLV,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("\nVision parity")
    tt_outputs.clear()
    hf_outputs.clear()
    vision_config = tt_config.vision_encoder
    if vision_config is None:
        raise ValueError("The Kimi K3 debug config must include a vision encoder.")
    patch_size = vision_config.patch_size
    grid_thw_N3 = torch.tensor([[1, 4, 4]], dtype=torch.long)
    patches_PCHW = torch.randn(
        16,
        3,
        patch_size,
        patch_size,
        dtype=dtype,
    )
    tt_grid_thw_N3 = grid_thw_N3.to(tt_device)
    hf_grid_thw_N3 = grid_thw_N3.to(hf_device)
    tt_patches_PCHW = patches_PCHW.to(tt_device)
    hf_patches_PCHW = patches_PCHW.to(hf_device)
    pixels_NPK = tt_patches_PCHW.reshape(1, 16, -1)
    tt_vision_NMD = tt_model.vision_encoder(
        pixels_NPK,
        grid_thw=tt_grid_thw_N3,
    )
    hf_vision = hf_model.vision_tower(hf_patches_PCHW, hf_grid_thw_N3)
    hf_vision_NMD = torch.stack(list(hf_model.mm_projector(hf_vision)))
    _compare_captured(
        tt_outputs,
        hf_outputs,
        atol=args.atol,
        rtol=args.rtol,
    )
    _compare_tensor(
        "vision.projected",
        tt_vision_NMD.float().cpu(),
        hf_vision_NMD.float().cpu(),
        atol=args.atol,
        rtol=args.rtol,
    )

    print("\nEnd-to-end image+text parity")
    tt_outputs.clear()
    hf_outputs.clear()
    num_vision_tokens = tt_vision_NMD.shape[1]
    hf_tokens_BL = torch.tensor(
        [[11, 23, _IMAGE_TOKEN_ID, 17, 31]],
        dtype=torch.long,
        device=hf_device,
    )
    tt_tokens_BL = torch.tensor(
        [[11, 23] + [_IMAGE_TOKEN_ID] * num_vision_tokens + [17, 31]],
        dtype=torch.long,
        device=tt_device,
    )
    hf_attention_mask_BL = torch.ones_like(hf_tokens_BL)
    tt_logits_BLV = tt_model(
        tt_tokens_BL,
        pixel_values=pixels_NPK,
        grid_thw=tt_grid_thw_N3,
        special_tokens={"image_id": _IMAGE_TOKEN_ID},
    )
    hf_logits_BLV = hf_model(
        input_ids=hf_tokens_BL,
        pixel_values=hf_patches_PCHW,
        grid_thws=hf_grid_thw_N3,
        attention_mask=hf_attention_mask_BL,
        use_cache=False,
        return_dict=True,
    ).logits
    _compare_captured(
        tt_outputs,
        hf_outputs,
        atol=args.atol,
        rtol=args.rtol,
    )
    _compare_tensor(
        "multimodal.logits",
        tt_logits_BLV.float().cpu(),
        hf_logits_BLV.float().cpu(),
        atol=args.atol,
        rtol=args.rtol,
    )
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
