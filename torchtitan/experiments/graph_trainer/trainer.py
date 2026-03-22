# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import DTensor

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_utils import CompiledModule
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config: Config):
        if config.compile.fake_tensors:
            self._fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            original_to_empty = torch.nn.Module.to_empty
            original_init_weights = CompiledModule.init_weights
            fake_mode = self._fake_mode

            def to_empty_under_fake_mode(module, *, device, **kwargs):
                with fake_mode:
                    return original_to_empty(module, device=device, **kwargs)

            torch.nn.Module.to_empty = to_empty_under_fake_mode
            CompiledModule.init_weights = lambda self, **kwargs: None
            try:
                super().__init__(config)
            finally:
                torch.nn.Module.to_empty = original_to_empty
                CompiledModule.init_weights = original_init_weights
        else:
            self._fake_mode = None
            super().__init__(config)

    def train(self):
        if self.config.compile.precompile and self._fake_mode is not None:
            self._precompile_with_fake_tensors()
        else:
            super().train()

    def _precompile_with_fake_tensors(self):
        """
        Trigger AOT compilation with fake tensors and save the artifact.
        We call the joint_graph_builder directly (not the full forward)
        because the compiled code cannot be executed with fake tensors
        (NCCL would try to communicate fake data). The on_compile
        callback in the builder automatically saves the artifact.
        """
        data_iterator = self.batch_generator(self.dataloader)
        input_dict, labels = next(data_iterator)

        with self._fake_mode:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels
            )

        # parallelize_inputs wraps fake tensors as DTensors via from_local(),
        # which only attaches placement metadata without allocating storage.
        # This runs outside FakeTensorMode so DTensor sees real mesh objects.
        if self.parallel_dims.pp_enabled:
            self._precompile_pp_stages(inputs, extra_inputs, extra_kwargs)
        else:
            model = self.model_parts[0]
            dt_args, dt_kwargs = model.parallelize_inputs(
                model.parallel_dims, (inputs,), {**extra_inputs, **extra_kwargs}
            )
            model.joint_graph_module = model.joint_graph_builder(
                model.inner, dt_args, dt_kwargs
            )

        logger.info(
            "Precompilation with fake tensors complete. "
            "Artifacts saved. Training cannot proceed with fake tensors."
        )

    def _precompile_pp_stages(self, inputs, extra_inputs, extra_kwargs):
        """
        Compile each pipeline parallelism stage in global order, propagating
        output shapes between ranks via dist.send_object_list/recv_object_list.
        Each stage is compiled independently by calling its joint_graph_builder
        with correctly-shaped fake tensor inputs. After compilation, we run
        the inner model under FakeTensorMode to compute output shapes for the
        next stage (FSDP/TP functional collectives dispatch to Meta
        implementations so no real communication occurs).
        """
        schedule = self.pp_schedule
        if hasattr(schedule, "_stages"):
            stages = schedule._stages
        else:
            stages = [schedule._stage]

        pp_group = stages[0].group
        my_pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        num_stages = stages[0].num_stages

        # Build stage_index → pp_rank mapping via all_gather_object
        my_stage_indices = [s.stage_index for s in stages]
        all_assignments: list[list[int] | None] = [None] * pp_size
        dist.all_gather_object(all_assignments, my_stage_indices, group=pp_group)

        stage_to_pp_rank: dict[int, int] = {}
        for pp_rank, indices in enumerate(all_assignments):
            for idx in indices:
                stage_to_pp_rank[idx] = pp_rank

        # Map local stage_index → (PipelineStage, CompiledModule)
        local_stages: dict[int, tuple] = {
            s.stage_index: (s, self.model_parts[i]) for i, s in enumerate(stages)
        }

        # grad_placements omitted: we are under torch.no_grad()
        # with fake tensors, so gradients are not relevant.
        def _to_local(t):
            if isinstance(t, DTensor):
                return t.to_local()
            return t

        # Iterate stages in global order. Each rank only compiles
        # stages it owns; the blocking send_object_list/recv_object_list
        # calls naturally pair up across ranks even for non-contiguous
        # schedules (e.g. V-schedule where rank 0 owns stages [0, 7])
        # because shape information flows sequentially from stage N to
        # N+1 and each rank's non-owned stages are skipped.
        prev_output = None
        for stage_idx in range(num_stages):
            if stage_idx not in local_stages:
                prev_output = None
                continue

            _stage, model = local_stages[stage_idx]

            if stage_idx == 0:
                dt_args, dt_kwargs = model.parallelize_inputs(
                    model.parallel_dims,
                    (inputs,),
                    {**extra_inputs, **extra_kwargs},
                )
            elif prev_output is not None:
                if isinstance(prev_output, torch.Tensor):
                    prev_output = (prev_output,)
                dt_args, dt_kwargs = model.parallelize_inputs(
                    model.parallel_dims, prev_output, {}
                )
            else:
                prev_pp_rank = stage_to_pp_rank[stage_idx - 1]
                shape_info: list = [None]
                dist.recv_object_list(
                    shape_info, group=pp_group, group_src=prev_pp_rank
                )
                with self._fake_mode:
                    fake_inputs = tuple(
                        torch.empty(shape, dtype=dtype, device=self.device)
                        for shape, dtype in shape_info[0]
                    )
                dt_args, dt_kwargs = model.parallelize_inputs(
                    model.parallel_dims, fake_inputs, {}
                )

            model.joint_graph_module = model.joint_graph_builder(
                model.inner, dt_args, dt_kwargs
            )

            if stage_idx + 1 < num_stages:
                with self._fake_mode, torch.no_grad():
                    output = model.inner(*dt_args, **dt_kwargs)

                output = torch.utils._pytree.tree_map(_to_local, output)

                next_pp_rank = stage_to_pp_rank[stage_idx + 1]
                if next_pp_rank == my_pp_rank:
                    prev_output = output
                else:
                    if isinstance(output, torch.Tensor):
                        shapes = [(tuple(output.shape), output.dtype)]
                    elif isinstance(output, (tuple, list)):
                        # Only tensor elements are propagated; non-tensor
                        # values (None, scalars) are dropped. This is safe
                        # because PP stage outputs are pure tensors today.
                        shapes = [
                            (tuple(t.shape), t.dtype)
                            for t in output
                            if isinstance(t, torch.Tensor)
                        ]
                    else:
                        raise TypeError(
                            f"Unexpected output type {type(output)} from "
                            f"stage {stage_idx}; expected Tensor or tuple"
                        )
                    dist.send_object_list(
                        [shapes], group=pp_group, group_dst=next_pp_rank
                    )
                    prev_output = None

        logger.info(f"Compiled {len(stages)} PP stage(s) on PP rank {my_pp_rank}")

    def close(self) -> None:
        super().close()

        # Note [explicit cudagraph close]
        # cudagraph holds reference to nccl which prevents destroy nccl
        # group. so we need to explicitly delete cudagraph which is held
        # in joint_graph_module. An explicit gc.collect() is necessary
        # to clean up reference cycles.
        for part in self.model_parts:
            if hasattr(part, "joint_graph_module"):
                part.joint_graph_module = None
        gc.collect()
