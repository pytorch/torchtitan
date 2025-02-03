# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


# This is similar to PrepareModuleInput and PrepareModuleOutput,
# but applies them simultaneously.
class PrepareModuleInputOutput(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, Tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_input: bool = True,
        output_layouts: Union[Placement, Tuple[Placement]],
        desired_output_layouts: Union[Placement, Tuple[Placement]],
        use_local_output: bool = True,
    ):
        # for input
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_input = use_local_input
        if self.input_layouts is not None:
            assert (
                self.desired_input_layouts is not None
            ), "desired module inputs should not be None!"
            assert len(self.input_layouts) == len(
                self.desired_input_layouts
            ), "input_layouts and desired_input_layouts should have same length!"
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            assert len(self.input_kwarg_layouts) == len(
                self.desired_input_kwarg_layouts
            ), "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"

        # for output
        self.output_layouts = (
            (output_layouts,)
            if isinstance(output_layouts, Placement)
            else output_layouts
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts
        )
        self.use_local_output = use_local_output
        assert len(self.output_layouts) == len(
            self.desired_output_layouts
        ), "output_layouts and desired_output_layouts should have same length!"

    def _prepare_input_arg(
        self,
        input: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(
                    input, torch.Tensor
                ), "expecting input to be a torch.Tensor!"
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_input else dt_inp
        else:
            return input

    def _prepare_input_fn(self, inputs, device_mesh):
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        assert (
            self.desired_input_layouts is not None
        ), "desired module inputs should not be None!"
        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key in kwarg_inputs.keys():
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError(
                "module outputs and output_layouts should have same length!"
            )
        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert out.placements[0] == out_layout
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,), run_check=False
                    )

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        else:
            return tuple(prepared_outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # for input
        if self.with_kwargs:
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )  # type: ignore[misc]
        else:
            module.register_forward_pre_hook(
                lambda _, inputs: self._prepare_input_fn(inputs, device_mesh)
            )  # type: ignore[misc, call-arg]

        # for output
        module.register_forward_hook(
            lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh)
        )  # type: ignore[misc, call-arg]

        return module


class TensorParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Partial(),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "gate_proj",
            nn.Parameter(distribute_tensor(module.gate_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding
        module.register_parameter(
            "down_proj",
            nn.Parameter(distribute_tensor(module.down_proj, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "up_proj",
            nn.Parameter(distribute_tensor(module.up_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class ExpertParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(0),)
        self.output_layouts = (output_layouts or Shard(0),)
        self.desired_input_layouts = (Shard(0),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ParallelStyle):
    def __init__(
        self,
        *,
        tp_mesh: DeviceMesh,
        ep_mesh: DeviceMesh,
    ):
        super().__init__()
        # TODO: has to pass in the meshes in addition to device_mesh,
        #       as there's an issue from DeviceMesh that
        #       "Cannot create a submesh from a submesh."
        self.tp_mesh = tp_mesh
        self.ep_mesh = ep_mesh

    @staticmethod
    def _prepare_input_fn(tp_mesh, ep_mesh, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        # input_tensor of placements Shard(1) on the tp mesh
        assert not isinstance(input_tensor, DTensor)

        # a2a(ep)
        input_tensor = DTensor.from_local(input_tensor, ep_mesh, (Shard(1),))
        input_tensor = input_tensor.redistribute(placements=(Shard(0),)).to_local()
        # ag(tp)
        input_tensor = DTensor.from_local(input_tensor, tp_mesh, (Shard(1),))
        input_tensor = input_tensor.redistribute(placements=(Replicate(),))

        return input_tensor

    @staticmethod
    def _partition_fn(tp_mesh, ep_mesh, name, module, device_mesh):
        # TODO: FSDP doesn't support sharding a 2D Tensor into a 3D one yet
        # module.register_parameter(
        #     "gate_proj",
        #     nn.Parameter(
        #         distribute_tensor(module.gate_proj, device_mesh, [Shard(0), Shard(2)])
        #     ),
        # )  # Column-wise sharding
        # module.register_parameter(
        #     "down_proj",
        #     nn.Parameter(
        #         distribute_tensor(module.down_proj, device_mesh, [Shard(0), Shard(1)])
        #     ),
        # )  # Row-wise sharding
        # module.register_parameter(
        #     "up_proj",
        #     nn.Parameter(
        #         distribute_tensor(module.up_proj, device_mesh, [Shard(0), Shard(2)])
        #     ),
        # )  # Column-wise sharding

        # TODO: Instead, for MoE experts, we shard on the EP mesh and then "forget" it.
        #       This would become an issue from DCP resharding perspective.
        module.register_parameter(
            "gate_proj",
            nn.Parameter(
                DTensor.from_local(
                    (
                        distribute_tensor(
                            module.gate_proj, device_mesh, [Shard(0), Shard(2)]
                        ).to_local()
                    ),
                    tp_mesh,
                    (Shard(2),),
                )
            ),
        )  # Column-wise sharding
        module.register_parameter(
            "down_proj",
            nn.Parameter(
                DTensor.from_local(
                    (
                        distribute_tensor(
                            module.down_proj, device_mesh, [Shard(0), Shard(1)]
                        ).to_local()
                    ),
                    tp_mesh,
                    (Shard(1),),
                )
            ),
        )  # Row-wise sharding
        module.register_parameter(
            "up_proj",
            nn.Parameter(
                DTensor.from_local(
                    (
                        distribute_tensor(
                            module.up_proj, device_mesh, [Shard(0), Shard(2)]
                        ).to_local()
                    ),
                    tp_mesh,
                    (Shard(2),),
                )
            ),
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(tp_mesh, ep_mesh, mod, outputs, device_mesh):
        # outputs of placements Partial() on the tp mesh

        # rs(tp)
        outputs = outputs.redistribute(placements=(Shard(1),)).to_local()
        # a2a(ep)
        outputs = DTensor.from_local(outputs, ep_mesh, (Shard(0),))
        outputs = outputs.redistribute(placements=(Shard(1),)).to_local()

        return outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partial(self._partition_fn, self.tp_mesh, self.ep_mesh),
            partial(self._prepare_input_fn, self.tp_mesh, self.ep_mesh),
            partial(self._prepare_output_fn, self.tp_mesh, self.ep_mesh),
        )
