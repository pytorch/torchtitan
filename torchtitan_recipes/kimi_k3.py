# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Kimi K3 recipes.

The MLA layers take a ``ContextParallelKernel`` through the generic transform
and the KDA layers take Attention Gym's context-parallel delta rule (KCP)
through :class:`KimiK3DeltaContextParallelTransform`; KDA is not an attention
config, so the generic transform and validation do not see it.
"""

from dataclasses import dataclass, fields
from typing import cast

from torchtitan.models.kimi_k3.context_parallel import (
    ContextParallelInnerKDA,
    MLAUlyssesCPFlexAttention,
)
from torchtitan.models.kimi_k3.kda import KDA
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer
from torchtitan.transforms import (
    apply_transforms,
    ContextParallelTransform,
    ModelTransform,
    retype_node,
)

__all__ = ["KimiK3DeltaContextParallelTransform", "kimi_k3_context_parallel"]


class KimiK3DeltaContextParallelTransform(ModelTransform):
    """Run every KDA layer on the context-parallel delta rule."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelTransform.Config):
        pass

    def transform(self, model: Module.Config) -> Module.Config:
        for _, traversed, _, _ in model.traverse(KDA.Config):
            kda = cast(KDA.Config, traversed)
            kda.inner_kda = retype_node(kda.inner_kda, ContextParallelInnerKDA)
        return model


def kimi_k3_context_parallel(
    config: Trainer.Config,
    *,
    cp_degree: int,
    mla_kernel: type[Module] = MLAUlyssesCPFlexAttention,
) -> Trainer.Config:
    """Kimi K3 under context parallelism: ``mla_kernel`` on the MLA layers,
    KCP on the KDA layers.

    Both read the sequence as rank-ordered contiguous chunks (Ulysses keeps the
    mask global and KCP hands its state from rank to rank), so the load balancer
    is off; the kernels take their group from the SPMD mesh, which only the
    ``spmd_types`` backend sets up.
    """
    config.parallelism.context_parallel_degree = cp_degree
    config.parallelism.context_parallel_load_balancer = None
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    model = config.model_spec.model
    assert isinstance(model, KimiK3Model.Config)
    attention = model.first_attention
    assert attention is not None, "Kimi K3 has full-attention layers."
    # The packed MLA kernels split the key on its rope slice; a generic kernel
    # (the upstream Ulysses or all-gather one) takes the expanded key as is.
    overrides: dict[str, object] = {}
    if "rope_head_dim" in {f.name for f in fields(mla_kernel.Config)}:
        overrides["rope_head_dim"] = getattr(attention, "qk_rope_head_dim")
    return apply_transforms(
        config,
        [
            ContextParallelTransform.Config(
                kernel=mla_kernel, kernel_config_overrides=overrides
            ),
            KimiK3DeltaContextParallelTransform.Config(),
        ],
    )
