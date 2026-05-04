# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel helpers for graph_trainer's ``aot_fx_trace`` path."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from autoparallel.api import AutoParallel
from autoparallel.input_validation import _check_forward_args, _compute_expected_inputs
from autoparallel.module_construction import make_parallel_module
from torch._functorch._aot_autograd.fx_utils import get_plain_input_and_grad_nodes
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class AutoParallelModelOutput:
    output_mesh: DeviceMesh
    output_placements: tuple
    sharded_output_axis: int


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


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _wrap_autoparallel_output(
    output: torch.Tensor,
    model_output: AutoParallelModelOutput | None,
) -> torch.Tensor:
    if model_output is None:
        return output
    output_shape = list(output.shape)
    output_shape[model_output.sharded_output_axis] *= model_output.output_mesh.size()
    output_shape = torch.Size(output_shape)
    return DTensor.from_local(
        output,
        device_mesh=model_output.output_mesh,
        placements=model_output.output_placements,
        run_check=False,
        shape=output_shape,
        stride=_contiguous_stride(output_shape),
    )


def _make_graph_trainer_compilers():
    def fw_compiler(gm: torch.fx.GraphModule, example_inputs):
        def run(args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(gm).boxed_run(args)

        run._boxed_call = True
        return run

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs):
        def run(args):
            with (
                fx_traceback.preserve_node_meta(),
                fx_traceback._set_autograd_backward(),
            ):
                return torch.fx.Interpreter(gm).boxed_run(args)

        run._boxed_call = True
        return run

    return fw_compiler, bw_compiler


class AutoParallelGraph(AutoParallel):
    """AutoParallel variant for graph_trainer's ``aot_fx_trace`` pipeline."""

    def apply_placement(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "graph_trainer AutoParallel no longer supports the deprecated AOT "
            "compiled path. Use apply_placement_for_fx_module()."
        )

    def apply_placement_for_fx_module(
        self,
        sharding_placement=None,
        *,
        model_output: AutoParallelModelOutput | None = None,
    ) -> nn.Module:
        """Return an AOT-backed parallel module for graph_trainer tracing.

        This keeps loss in graph_trainer's normal train step. The optional output
        adapter is only needed when the local AutoParallel output must re-enter
        PyTorch as a DTensor, e.g. vocab-sharded Llama logits for loss_parallel().
        """
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )
        if sharding_placement is None:
            sharding_placement = self.sharding_placement
        fw_compiler, bw_compiler = _make_graph_trainer_compilers()
        parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )

        graph_param_fqns = list(self.joint_with_descriptors.params_spec)
        graph_buffer_fqns = list(self.joint_with_descriptors.buffers_spec)

        input_nodes = get_plain_input_and_grad_nodes(self.gm.graph)
        solved_input_placements = []
        for desc in sorted(input_nodes, key=lambda input_desc: input_desc.idx):
            node, _grad_node = input_nodes[desc]
            strategy = sharding_placement[node]
            solved_input_placements.append(tuple(strategy.output_specs.placements))
        expected_inputs = _compute_expected_inputs(
            self._traced_inputs,
            solved_input_placements,
            self.mesh,
        )

        def forward(self, *args):
            flat_args, _ = torch.utils._pytree.tree_flatten(args)
            _check_forward_args(flat_args, expected_inputs)
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
            boxed_args = [*params, *flat_args]
            del params
            output = parallel_model_fn(boxed_args)
            return _wrap_autoparallel_output(output, model_output)

        return make_parallel_module(
            self.model,
            sharded_param_dict,
            sharded_buffer_dict,
            forward_fn=forward,
        )
