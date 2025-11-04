# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import torch
from torchtitan.config import JobConfig


def get_compile_backend(job_config: JobConfig) -> Union[str, callable]:
    # return the compile backends used in SimpleFSDP training
    # Step1: check if backend_name is inside available torch.compile backends
    # Step2: check if the backend_name has been registered as a customized backend
    backend_name = (
        getattr(job_config.compile, "model_backend_override", None)
        or job_config.compile.backend
    )

    available_torch_backend = torch._dynamo.list_backends(exclude_tags=())
    if backend_name in available_torch_backend:
        return backend_name

    if backend_name == "aot_eager_autobucketing":
        # Perform auto optimization in aten fx-level and execute code in aot_eager backend
        # The autobucketing logic is here: https://github.com/pytorch/pytorch/pull/163960
        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        torch._inductor.config.test_configs.aten_fx_overlap_preserving_bucketing = True
        torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps = False
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
    elif backend_name == "aot_eager_manualbucketing":
        # Perform manual optimization in aten fx-level and execute code in aot_eager backend
        # The manualbucketing logic is here:
        bucketing_modules = job_config.compile.manual_bucketed_modules
        from functools import partial

        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
        from torch._inductor.fx_passes.overlap_manual_scheduling import (
            manual_overlap_bucketing,
        )

        torch._inductor.config.allow_buffer_reuse = False
        manual_overlap_bucketing = partial(
            manual_overlap_bucketing,
            module_bucket_plans=job_config.compile.manual_bucketed_modules,
        )

        def aten_manualbucketing_reordering_pass(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> torch.fx.GraphModule:
            manual_overlap_bucketing(gm)
            return gm

        backend = aot_autograd_backend(
            fw_compiler=aten_manualbucketing_reordering_pass,
            bw_compiler=aten_manualbucketing_reordering_pass,
            keep_inference_input_mutations=True,
        )
    else:
        raise AssertionError(f"Unsupported customized backend: {backend_name}")

    return backend
