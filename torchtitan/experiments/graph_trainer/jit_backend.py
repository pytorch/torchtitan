# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch._functorch.config as functorch_config

from torchtitan.tools.logging import logger

from .configs import GraphTrainerCompileConfig as CompileConfig
from .passes import (
    autobucketing_reordering_pass,
    fsdp_reshard_after_fwd_pass,
    transformer_block_bucketing_reordering_pass,
)


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

    # Resolve JIT pass name from the passes list
    jit_pass_name = None
    if compile_config.passes:
        if len(compile_config.passes) != 1 or compile_config.passes[0] not in {
            "auto_bucketing",
            "transformer_block_bucketing",
        }:
            raise ValueError(
                "JIT mode currently supports at most one pass, which should be "
                "either 'auto_bucketing' or 'transformer_block_bucketing', "
                f"got: {compile_config.passes}"
            )
        jit_pass_name = compile_config.passes[0]

    # Apply bucketing and overlapping pass on fwd and bwd graph separately
    if jit_pass_name == "auto_bucketing":
        # Perform auto optimization in aten fx-level and execute code in aot_eager/inductor backend
        # The autobucketing logic is here: https://github.com/pytorch/pytorch/pull/163960
        from torch._inductor.config import aten_distributed_optimizations as dist_opts
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing_from_inductor_configs,
        )

        if compile_config.backend == "aot_eager":
            from torch._dynamo.backends.common import (
                aot_autograd as aot_autograd_backend,
            )

            dist_opts.insert_overlap_deps = False
            backend = aot_autograd_backend(
                fw_compiler=autobucketing_reordering_pass,
                bw_compiler=autobucketing_reordering_pass,
                keep_inference_input_mutations=True,
            )
        elif compile_config.backend == "inductor":

            def inductor_autobucketing_reordering_pass(
                gm: torch.fx.Graph,
            ) -> torch.fx.GraphModule:
                return schedule_overlap_bucketing_from_inductor_configs(
                    gm.owning_module
                )

            dist_opts.collective_bucketing = True
            dist_opts.insert_overlap_deps = True
            torch._inductor.config.allow_buffer_reuse = False
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

    elif jit_pass_name == "transformer_block_bucketing":
        # Perform manual optimization in aten fx-level and execute code in aot_eager/inductor backend
        # The manualbucketing logic is here: https://github.com/pytorch/pytorch/pull/165487
        from functools import partial

        from torch._inductor.fx_passes.overlap_manual_scheduling import (
            manual_overlap_bucketing,
        )

        if compile_config.backend == "aot_eager":
            from torch._dynamo.backends.common import (
                aot_autograd as aot_autograd_backend,
            )

            backend = aot_autograd_backend(
                fw_compiler=partial(
                    transformer_block_bucketing_reordering_pass,
                    fsdp_manual_buckets=fsdp_manual_buckets,
                ),
                bw_compiler=partial(
                    transformer_block_bucketing_reordering_pass,
                    fsdp_manual_buckets=fsdp_manual_buckets,
                ),
                keep_inference_input_mutations=True,
            )
        elif compile_config.backend == "inductor":

            def inductor_transformer_block_bucketing_reordering_pass(
                gm: torch.fx.Graph,
            ) -> torch.fx.GraphModule:
                return manual_overlap_bucketing(
                    gm.owning_module,
                    module_bucket_plans=fsdp_manual_buckets,
                    insert_overlap_deps=True,
                )

            torch._inductor.config.allow_buffer_reuse = False
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

    def joint_ac_pass(
        gm: torch.fx.GraphModule, example_inputs: Any
    ) -> torch.fx.GraphModule:
        return fsdp_reshard_after_fwd_pass(
            gm, example_inputs, reshard_after_forward=fsdp_reshard_after_forward
        )

    def graph_trainer_custom_pass(*args, **kwargs):
        # the ac pass has to operate in a joint graph before partitioner for ac
        # annotation to take into effect.
        with functorch_config.patch("joint_custom_pass", joint_ac_pass):
            return backend(*args, **kwargs)

    return graph_trainer_custom_pass
