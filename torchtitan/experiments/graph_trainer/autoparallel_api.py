# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel helpers for graph_trainer's ``aot_fx_trace`` path."""

import torch
import torch.nn as nn
from autoparallel.api import AutoParallel
from autoparallel.module_construction import make_parallel_module
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch.distributed.tensor import DTensor

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


def _local_tensor_with_autograd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _get_raw_module_tensor(
    module: nn.Module, fqn: str, *, is_buffer: bool
) -> torch.Tensor:
    *prefix, name = fqn.split(".")
    owner = module.get_submodule(".".join(prefix)) if prefix else module
    tensor_dict = owner._buffers if is_buffer else owner._parameters
    tensor = tensor_dict.get(name)
    if tensor is None:
        kind = "buffer" if is_buffer else "parameter"
        raise AttributeError(f"{fqn!r} is not a registered {kind}")
    return tensor


class AutoParallelGraph(AutoParallel):
    """AutoParallel variant for graph_trainer's ``aot_fx_trace`` pipeline."""

    def apply_placement_for_fx_module(
        self,
        sharding_placement=None,
        *,
        compile_config: GraphTrainerCompileConfig,
    ) -> nn.Module:
        """Return an AOT-backed parallel module for graph_trainer tracing.

        This keeps loss in graph_trainer's normal train step. The AOT callable
        owns input-tree flattening and keyword ordering, and the module returns
        its local output, e.g. vocab-sharded Llama logits for the TP loss.
        """
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )
        parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors,
            fw_compiler=self.compiler_fn,
            bw_compiler=self.compiler_fn,
        )

        graph_param_fqns = list(self.joint_with_descriptors.params_spec)
        graph_buffer_fqns = list(self.joint_with_descriptors.buffers_spec)

        def forward(self, *args, **kwargs):
            params = [
                _local_tensor_with_autograd(
                    _get_raw_module_tensor(self, fqn, is_buffer=False)
                )
                for fqn in graph_param_fqns
            ] + [
                _local_tensor_with_autograd(
                    _get_raw_module_tensor(self, fqn, is_buffer=True)
                )
                for fqn in graph_buffer_fqns
            ]
            output = parallel_model_fn(*params, *args, **kwargs)
            del params
            return output

        return make_parallel_module(
            self.model,
            sharded_param_dict,
            sharded_buffer_dict,
            forward_fn=forward,
        )
