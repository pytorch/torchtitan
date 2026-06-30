# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from xx.common.compressor_helpers import COMPRESSOR_STATS

from torchtitan.components.optimizer import register_float8_precompute_scale_hook
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import parallelize_worldmodel, TransformerConfig, WorldModel


COMPRESSOR_MODEL = "c04337f8-b83f-4e34-b07a-5f7396978d67"
LATENT_CHANNELS = 32
LATENT_SIZE = (16, 32)

WORLD_MODEL_FLOAT8_FILTER_FQNS = [
    "x_embedder",
    "augments_pos_ref_augment_embedder",
    "ref_augment_from_augments_euler_embedder",
    "pose_mask_embedder",
    "t_embedder",
    "fidx_embedder",
    "final_layer",
    "plan_head",
]


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = _worldmodel_configs()[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="worldmodel",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_worldmodel,
        pipelining_fn=None,
        post_optimizer_build_fn=register_float8_precompute_scale_hook,
        state_dict_adapter=None,
    )


def _worldmodel_configs() -> dict[str, Callable[[], WorldModel.Config]]:
    return {
        "base": _model_config,
        "debugmodel": _debug_model_config,
    }


def _debug_model_config() -> WorldModel.Config:
    return _model_config(
        input_size=(15, 4, 4),
        patch_size=(1, 2, 2),
        hidden=128,
        heads=4,
        layers=1,
        plan_layers=1,
        mlp_multiple_of=16,
        attention_impl="FLEX",
    )


def _model_config(
    *,
    input_size: tuple[int, int, int] = (15, *LATENT_SIZE),
    patch_size: tuple[int, int, int] = (1, 2, 2),
    hidden: int = 2304,
    heads: int = 36,
    layers: int = 56,
    plan_layers: int = 4,
    mlp_multiple_of: int = 256,
    attention_impl: str = "FLEX",
    attention_mask: str = "LAST_FRAME_CAUSAL",
    norm: str = "RMSNorm",
) -> WorldModel.Config:
    stats = COMPRESSOR_STATS[COMPRESSOR_MODEL]
    return WorldModel.Config(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        pose_size=6,
        time_factor=1.0,
        compressor_mean=stats["mean"],
        compressor_std=stats["std"],
        transformer=TransformerConfig(
            n_layer=layers,
            n_embd=hidden,
            n_head=heads,
            act="GELU",
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            biased_linears=True,
            prenorm=False,
            qk_norm=True,
            mlp_mult=4,
            mlp_multiple_of=mlp_multiple_of,
            attention_mask=attention_mask,
            norm=norm,
            attention_impl=attention_impl,
        ),
        plan_head=TransformerConfig(
            n_layer=plan_layers,
            n_embd=hidden,
            n_head=heads,
            act="GELU",
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            biased_linears=True,
            prenorm=True,
            qk_norm=False,
            mlp_mult=2,
            mlp_multiple_of=1,
            attention_mask=attention_mask,
            norm="LayerNorm",
            attention_impl=attention_impl,
        ),
    )


def _blocks_only_float8(
    *, model_compile_enabled: bool, emulate: bool = False
) -> Float8LinearConverter.Config:
    return Float8LinearConverter.Config(
        recipe_name="tensorwise",
        filter_fqns=WORLD_MODEL_FLOAT8_FILTER_FQNS,
        emulate=emulate,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        model_compile_enabled=model_compile_enabled,
    )


def main() -> None:
    spec = model_registry("debugmodel")
    model = spec.model.build()
    config = model.config
    nparams = sum(param.numel() for param in model.parameters())
    head_dim = config.transformer.n_embd // config.transformer.n_head
    print(
        {
            "flavor": spec.flavor,
            "input_size": config.input_size,
            "num_patches": config.num_patches,
            "hidden": config.transformer.n_embd,
            "heads": config.transformer.n_head,
            "head_dim": head_dim,
            "layers": config.transformer.n_layer,
            "parameters": nparams,
        }
    )


if __name__ == "__main__":
    main()
