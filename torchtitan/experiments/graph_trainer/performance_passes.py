# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph passes that improve performance but may change numerics.

Gated behind ``--compile.numerics_changing_optim`` (opt-in, default off).
"""

import operator

import torch

from torchtitan.tools.logging import logger


def annotate_rmsnorm_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    rmsnorm_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag RMSNorm ops with compile_with_inductor for regional_inductor.

    Identifies ``_fused_rms_norm`` and ``_fused_rms_norm_backward`` nodes
    by their ``node.target`` and tags them (along with their ``getitem``
    users) so that ``regional_inductor_pass`` compiles each norm as a
    fused Inductor kernel.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        rmsnorm_compile_config: Inductor config dict for RMSNorm nodes.
            When provided, wrapped as ``{"inductor_configs": rmsnorm_compile_config}``.
            Default is None (no inductor_configs in the annotation).
    """
    compile_annotation: dict = (
        {"inductor_configs": rmsnorm_compile_config}
        if rmsnorm_compile_config is not None
        else {}
    )

    _RMSNORM_TARGETS = {
        torch.ops.aten._fused_rms_norm.default,
        torch.ops.aten._fused_rms_norm_backward.default,
    }

    num_tagged = 0

    for node in gm.graph.nodes:
        if node.target not in _RMSNORM_TARGETS:
            continue

        node.meta.setdefault("custom", {})["compile_with_inductor"] = compile_annotation
        num_tagged += 1

        # Tag getitem users that extract outputs from the fused op.
        for user in node.users:
            if user.target is operator.getitem:
                user.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = compile_annotation

    if num_tagged > 0:
        logger.info(
            f"Tagged {num_tagged} RMSNorm nodes for regional Inductor compilation"
        )

    return gm
