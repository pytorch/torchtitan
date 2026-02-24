# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch._functorch.config as functorch_config
from torchtitan.tools.logging import logger

from .configs import SimpleFSDPCompileConfig as CompileConfig

from .reshard_after_forward import annotate_fsdp_all_gather


def get_compile_backend_with_passes(
    compile_config: CompileConfig,
    fsdp_reshard_after_forward: bool,
    fsdp_manual_buckets: list[list[str] | str] | None,
) -> callable:
    """
    Apply compile backend and additional graph passes.
    Args:
        compile_config: compile configs to apply torch.compile.
        fsdp_reshard_after_forward: whether to enable reshard_after_forward in SimpleFSDP,
            which is implemented via a customized AC graph pass.
        fsdp_manual_buckets: used in transformer_block_bucketing to define which modules should be bucketed.
    Returns:
        compile backend with applied graph passes.
    """
    backend = torch._dynamo.lookup_backend(compile_config.backend)

    # Apply bucketing and overlapping pass on fwd and bwd graph separately
    if compile_config.graph_passes == "auto_bucketing":
        # Perform auto optimization in aten fx-level and execute code in aot_eager/inductor backend
        # The autobucketing logic is here: https://github.com/pytorch/pytorch/pull/163960
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

    elif compile_config.graph_passes == "transformer_block_bucketing":
        # Perform manual optimization in aten fx-level and execute code in aot_eager/inductor backend
        # The manualbucketing logic is here: https://github.com/pytorch/pytorch/pull/165487
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
        # when fsdp_reshard_after_forward set to True, it will annotate simple_fsdp AG
        #   to CheckpointPolicy.MUST_RECOMPUTE.
        # when fsdp_reshard_after_forward set to False, it will annotate simple_fsdp AG
        #   to CheckpointPolicy.MUST_SAVE.
        gm = annotate_fsdp_all_gather(gm, fsdp_reshard_after_forward)
        gm.recompile()
        return gm

    def simple_fsdp_custom_pass(*args, **kwargs):
        # the ac pass has to operate in a joint graph before partitioner for ac
        # annotation to take into effect.
        with functorch_config.patch("joint_custom_pass", joint_ac_pass):
            return backend(*args, **kwargs)

    return simple_fsdp_custom_pass
