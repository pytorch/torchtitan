# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inductor lite pass for the compiler toolkit.

This module provides inductor lite pass that can be applied to graph modules
during compilation.
"""
from typing import Optional

import torch
from torchtitan.tools.logging import logger


def get_inductor_lite_fw_compiler(extra_config: Optional[dict] = None):
    from torch._inductor import lite_mode_options
    from torch._inductor.compile_fx import compile_fx_inner

    context = torch._guards.TracingContext.try_get()

    if not context or not context.fw_metadata:
        logger.warn("No context or fw_metadata available")
        static_input_idxs = ()
    else:
        static_input_idxs = context.fw_metadata.static_input_indices

    inductor_config = lite_mode_options
    if extra_config:
        inductor_config.update(extra_config)

    def fw_compiler(gm: torch.fx.GraphModule, example_inputs: tuple):
        with torch._inductor.config.patch(inductor_config):
            compiled_fn = compile_fx_inner(
                gm,
                example_inputs,
                static_input_idxs=static_input_idxs,
                is_backward=False,
            )
        return compiled_fn

    return fw_compiler


def get_inductor_lite_bw_compiler(extra_config: Optional[dict] = None):
    from torch._inductor import lite_mode_options
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.utils import count_tangents

    inductor_config = lite_mode_options
    if extra_config:
        inductor_config.update(extra_config)

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: tuple):
        fixed = count_tangents(gm)

        with torch._inductor.config.patch(inductor_config):
            compiled_fn = compile_fx_inner(
                gm,
                example_inputs,
                static_input_idxs=list(range(fixed)),
                is_backward=True,
            )
        return compiled_fn

    return bw_compiler
