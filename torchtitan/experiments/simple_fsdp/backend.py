# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch._functorch.config as functorch_config

from .reshard_after_forward import annotate_fsdp_all_gather


def get_compile_backend(
    backend_name: str, fsdp_reshard_after_forward: bool
) -> callable:
    # return the compile backends used in SimpleFSDP training
    # Step1: check if backend_name is inside available torch.compile backends
    # Step2: check if the backend_name has been registered as a customized backend
    available_torch_backend = torch._dynamo.list_backends(exclude_tags=())

    if backend_name in available_torch_backend:
        backend = torch._dynamo.lookup_backend(backend_name)
    elif backend_name == "aot_eager_autobucketing":
        # Perform auto optimization in aten fx-level and execute code in aot_eager backend
        # The autobucketing logic is here: https://github.com/pytorch/pytorch/pull/163960
        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend

        from torch._inductor.config import aten_distributed_optimizations as dist_opts
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        dist_opts.collective_bucketing = True
        dist_opts.insert_overlap_deps = False
        torch._inductor.config.allow_buffer_reuse = False

        def aten_autobucketing_reordering_pass(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> torch.fx.GraphModule:
            schedule_overlap_bucketing(gm)
            gm.recompile()
            return gm

        backend = aot_autograd_backend(
            fw_compiler=aten_autobucketing_reordering_pass,
            bw_compiler=aten_autobucketing_reordering_pass,
            keep_inference_input_mutations=True,
        )
    else:
        raise AssertionError(f"Unsupported customized backend: {backend_name}")

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
