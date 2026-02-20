# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JIT compilation backend for graph-based training.

This module provides a custom torch.compile backend that integrates
graph-level optimization passes (bucketing, overlapping, activation
checkpointing annotation) with Simple FSDP.
"""

from typing import Any

import torch
import torch._functorch.config as functorch_config

from torchtitan.tools.logging import logger

from .reshard_after_forward import annotate_fsdp_all_gather


def get_jit_compile_backend(
    compile_config,
    fsdp_reshard_after_forward: bool,
    post_partition_pass_names: list[str],
    fsdp_manual_buckets: list[list[str] | str] | None,
) -> callable:
    """
    Build a torch.compile backend with the given passes for JIT mode.

    Args:
        compile_config: The compile section of job config (has .backend field)
        fsdp_reshard_after_forward: Whether to reshard after forward
        post_partition_pass_names: List of post-partition pass names to apply
        fsdp_manual_buckets: Bucket plans for transformer_block_bucketing

    Returns:
        A callable backend for torch.compile
    """
    backend = torch._dynamo.lookup_backend(compile_config.backend)

    # Determine which bucketing pass to apply (at most one)
    bucketing_pass = None
    for name in post_partition_pass_names:
        if name in ("auto_bucketing", "transformer_block_bucketing"):
            bucketing_pass = name

    if bucketing_pass == "auto_bucketing":
        from torch._inductor.config import aten_distributed_optimizations as dist_opts
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        dist_opts.collective_bucketing = True
        torch._inductor.config.allow_buffer_reuse = False

        if compile_config.backend == "aot_eager":
            from torch._dynamo.backends.common import (
                aot_autograd as aot_autograd_backend,
            )

            def aot_eager_autobucketing_reordering_pass(
                gm: torch.fx.GraphModule, example_inputs: Any
            ) -> torch.fx.GraphModule:
                schedule_overlap_bucketing(gm)
                gm.recompile()
                return gm

            dist_opts.insert_overlap_deps = False
            backend = aot_autograd_backend(
                fw_compiler=aot_eager_autobucketing_reordering_pass,
                bw_compiler=aot_eager_autobucketing_reordering_pass,
                keep_inference_input_mutations=True,
            )
        elif compile_config.backend == "inductor":

            def inductor_autobucketing_reordering_pass(
                gm: torch.fx.Graph,
            ) -> torch.fx.GraphModule:
                return schedule_overlap_bucketing(gm.owning_module)

            dist_opts.insert_overlap_deps = True
            torch._inductor.config.reorder_for_peak_memory = False
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.post_grad_custom_post_pass = (
                inductor_autobucketing_reordering_pass
            )
        else:
            raise ValueError(
                f"Unsupported backend {compile_config.backend} for auto_bucketing pass"
            )
        logger.info("Auto bucketing pass is applied")

    elif bucketing_pass == "transformer_block_bucketing":
        from functools import partial

        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
        from torch._inductor.fx_passes.overlap_manual_scheduling import (
            manual_overlap_bucketing,
        )

        torch._inductor.config.allow_buffer_reuse = False
        manual_overlap_bucketing = partial(
            manual_overlap_bucketing,
            module_bucket_plans=fsdp_manual_buckets,
        )

        if compile_config.backend == "aot_eager":

            def aot_eager_transformer_block_bucketing_reordering_pass(
                gm: torch.fx.GraphModule, example_inputs: Any
            ) -> torch.fx.GraphModule:
                manual_overlap_bucketing(gm, insert_overlap_deps=False)
                return gm

            backend = aot_autograd_backend(
                fw_compiler=aot_eager_transformer_block_bucketing_reordering_pass,
                bw_compiler=aot_eager_transformer_block_bucketing_reordering_pass,
                keep_inference_input_mutations=True,
            )
        elif compile_config.backend == "inductor":

            def inductor_transformer_block_bucketing_reordering_pass(
                gm: torch.fx.Graph,
            ) -> torch.fx.GraphModule:
                return manual_overlap_bucketing(
                    gm.owning_module, insert_overlap_deps=True
                )

            torch._inductor.config.reorder_for_peak_memory = False
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.post_grad_custom_post_pass = (
                inductor_transformer_block_bucketing_reordering_pass
            )
        else:
            raise ValueError(
                f"Unsupported backend {compile_config.backend} for transformer_block_bucketing pass"
            )
        logger.info("Transformer block bucketing pass is applied")

    else:
        logger.info("No bucketing or overlapping pass is applied")

    # Apply activation checkpointing on joint graph before partitioner
    def joint_ac_pass(
        gm: torch.fx.GraphModule, example_inputs: Any
    ) -> torch.fx.GraphModule:
        # this pass implements simplefsdp's fsdp_reshard_after_forward behavior
        gm = annotate_fsdp_all_gather(gm, fsdp_reshard_after_forward)
        gm.recompile()
        return gm

    def simple_fsdp_custom_pass(*args, **kwargs):
        # the ac pass has to operate in a joint graph before partitioner for ac
        # annotation to take into effect.
        with functorch_config.patch("joint_custom_pass", joint_ac_pass):
            return backend(*args, **kwargs)

    return simple_fsdp_custom_pass
