# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, List

import torch.nn as nn

from torchtitan.config_manager import JobConfig, MX
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn

# Maps titan recipe names to torchao mx recipe names
NAME_MAP = {"mxfp8": "mxfp8_cublas"}


class MXConverter(ModelConverter):
    """Converts the linear layers of `model` to `MXLinear`."""

    enabled: bool
    filter_fqns: List[str]
    mx_config: Any  # MXLinearConfig type when imported

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        # Ensure minimum torchao versions
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )
        torchao_version = version("torchao")
        mxfp8_min_version = "0.11.0"
        if torchao_version < mxfp8_min_version:
            raise ImportError(
                f"torchao version {torchao_version} is too old, please install torchao {mxfp8_min_version} or later and try again"
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or architectures"

        self.enabled = True
        mx_job_config: MX = job_config.mx
        self.filter_fqns = mx_job_config.filter_fqns

        # Configure MXFP8
        from torchao.prototype.mx_formats.config import MXLinearConfig

        config = MXLinearConfig.from_recipe_name(NAME_MAP[mx_job_config.recipe_name])
        config.use_fp8_dim1_cast_triton_kernel = (
            mx_job_config.use_fp8_dim1_cast_triton_kernel
        )
        self.config = config

        logger.info(f"Float8 training active with recipe {mx_job_config.recipe_name}")

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `MXLinear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.prototype.mx_formats.config import MXLinearConfig
        from torchao.quantization import quantize_

        assert isinstance(self.config, MXLinearConfig)
        quantize_(
            model,
            config=self.config,
            filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
        )
        logger.info("Swapped to MXLinear layers")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 doesn't require any post-optimizer hooks at the moment
        """
        return


register_model_converter(MXConverter, "mx")
