# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance

from typing import Callable, List, Union

import torch
import torch.nn as nn

from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger
from torchtitan.parallelisms import ParallelDims


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


class Float8Handler:
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = False

        float8_config = job_config.float8
        if not float8_config.enable_float8_linear:
            return
        if not _is_sm89_or_later():
            logger.warning(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        self.use_float8nocompile = float8_config.float8nocompile
        self.ac_config = job_config.activation_checkpoint

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = (
            parallel_dims.dp_shard_enabled
            and float8_config.enable_fsdp_float8_all_gather
        )
        scaling_type_input = ScalingType(float8_config.scaling_type_input)
        scaling_type_weight = ScalingType(float8_config.scaling_type_weight)
        scaling_type_grad_output = ScalingType(float8_config.scaling_type_grad_output)
        self.config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        )

        self.enabled = True

        # for precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = (
            enable_fsdp_float8_all_gather
            and float8_config.precompute_float8_dynamic_scale_for_fsdp
        )

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input is ScalingType.DELAYED
            or scaling_type_weight is ScalingType.DELAYED
            or scaling_type_grad_output is ScalingType.DELAYED
        )
        self._sync_float8_amax_and_scale_history = None
        self.compile = job_config.training.compile

        logger.info("Float8 training active")

    def convert_to_float8_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        if self.use_float8nocompile:
            logger.info("Using float8nocompile prototype")
            from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
                convert_to_float8_nocompile_training,
            )

            # for full AC or no AC
            no_precompute_for_backward = self.ac_config.mode == "full"
            convert_to_float8_nocompile_training(
                model,
                config=self.config,
                module_filter_fn=lambda mod, fqn: fqn != "output",
                no_precompute_for_backward=no_precompute_for_backward,
            )

            # for selective per layer AC
            if (
                self.ac_config.mode == "selective"
                and self.ac_config.selective_ac_option.isdigit()
            ):
                no_precompute_for_backward_every_nth_layer(
                    model,
                    int(self.ac_config.selective_ac_option),
                )
        else:
            logger.info("Using float8 training")
            from torchao.float8 import convert_to_float8_training

            # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
            convert_to_float8_training(
                model,
                config=self.config,
                module_filter_fn=lambda mod, fqn: fqn != "output",
            )
        logger.info(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history

        # TODO(vkuzo): see if precalculating the modules to sync over is going to
        # meaningfully help performance

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = (
                    sync_float8_amax_and_scale_history
                )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)


def no_precompute_for_backward_every_nth_layer(model: nn.Module, n: int):
    """Set no_precompute_for_backward to True for every nth layer in the model."""
    for layer_idx, (layer_id, layer) in enumerate(model.layers.named_children()):
        if layer_idx % n == 0:
            logger.info(f"Enabling no_precompute_for_backward for layer {layer_id}")
            _enable_no_precompute_for_backward(layer)


def _enable_no_precompute_for_backward(model: nn.Module):
    """Recursively set no_precompute_for_backward to True for all linear layers in the given model."""
    for child_layer in model.children():
        if isinstance(child_layer, nn.Linear):
            child_layer.no_precompute_for_backward = True
        else:
            _enable_no_precompute_for_backward(child_layer)
