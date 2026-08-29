# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)
from torchtitan.tools.logging import logger

from .kda import KimiDeltaAttention
from .model import KimiK3Model, KimiMLAAttention


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Apply FSDP2 and context parallelism to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (
            ("tensor parallel", parallel_dims.tp_enabled),
            ("pipeline parallel", parallel_dims.pp_enabled),
            ("expert parallel", parallel_dims.ep_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 currently supports FSDP2 data parallelism and context "
            f"parallelism only; disable {', '.join(unsupported_parallelisms)}."
        )
    if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(
            "Kimi K3 FSDP2 currently supports the partial_dtensor SPMD backend "
            "only; the config registry pins it."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    assert isinstance(model, KimiK3Model)
    if parallel_dims.cp_enabled:
        apply_cp_kimi_k3(model, parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    vision_encoder = model.vision_encoder
    if vision_encoder is not None:
        # TODO: An image batch on one DP rank and a text-only batch on another
        # execute different FSDP collectives, deadlock, and hit a 90-second
        # timeout. A general solution is needed.
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=1,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


def apply_cp_kimi_k3(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Wire context parallelism: KCP on the KDA layers, Ulysses on the MLA layers.

    Both at once, on disjoint layer kinds. Imperative rather than declared:
    KDA's kernels are fla triton and never see a DTensor, so no ShardingConfig
    can drive them (the model config overrides ``_validate_cp_backend`` for the
    same reason).
    """
    cp_group = parallel_dims.get_mesh("cp").get_group()
    cp_degree = parallel_dims.cp
    model._cp_group = cp_group

    num_mla = 0
    kda_modules = []
    for module in model.modules():
        if isinstance(module, KimiMLAAttention):
            if module.n_heads % cp_degree != 0:
                raise ValueError(
                    f"MLA n_heads={module.n_heads} must be divisible by "
                    f"cp={cp_degree} for Ulysses head sharding"
                )
            module._cp_group = cp_group
            num_mla += 1
        elif isinstance(module, KimiDeltaAttention):
            kda_modules.append(module)

    if kda_modules:
        # Checked at wiring time so the message is actionable, rather than an
        # ImportError from inside a layer's first forward.
        try:
            from fla.modules.conv.cp.ops import causal_conv1d_cp  # noqa: F401
            from fla.ops.cp.context import build_cp_context  # noqa: F401
        except ImportError as err:
            raise ValueError(
                "KDA context parallelism needs fla-core's CP ops "
                "(fla.ops.cp.context.build_cp_context and "
                "fla.modules.conv.cp.ops.causal_conv1d_cp), which ship in "
                f"fla-core >= 0.5.1; import failed with: {err}."
            ) from err

    for module in kda_modules:
        module._cp_group = cp_group
    if num_mla + len(kda_modules) == 0:
        raise ValueError(
            "context parallel is enabled but no attention layer was found to "
            "wire it onto."
        )
    logger.info(
        "Applied context parallel to %d MLA and %d KDA layer(s).",
        num_mla,
        len(kda_modules),
    )
