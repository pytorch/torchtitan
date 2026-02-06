# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch
import torch.distributed.checkpoint.stateful

from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.ft.manager import _FT_MANAGER
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.train import main, Trainer


class FTTrainer(Trainer):
    ft_manager: FTManager | None = None

    def setup_ft_manager(self) -> FTManager:
        """Initialize the FT manager if it is not already initialized."""
        if self.ft_manager is None:
            self.ft_manager = _FT_MANAGER
            self.ft_manager.setup(self.job_config.fault_tolerance)
        return self.ft_manager

    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(
                    self.model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.setup_ft_manager(),
        )

        self.setup_ft_manager().maybe_set_all_reduce_hook(self.model_parts)

    def init_distributed(self) -> ParallelDims:
        job_config = self.job_config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = job_config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
            ranks=global_ranks,
        )

        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism

        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def get_dp_info(self) -> tuple[int, int]:
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        return self.setup_ft_manager().get_dp_info(batch_degree, batch_rank)

    def compute_global_losses(
        self,
        parallel_dims: ParallelDims,
        loss: torch.Tensor,
        global_valid_tokens: float,
        local_valid_tokens: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Compute the global loss across all ranks."""
        if parallel_dims.dp_cp_enabled:
            pg = self.setup_ft_manager().loss_sync_pg
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            return (
                dist_utils.dist_sum(loss, loss_mesh, pg),
                dist_utils.dist_max(local_avg_loss, loss_mesh, pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen
            return global_avg_loss, global_max_loss, global_ntokens_seen

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        ft_manager = self.setup_ft_manager()

        leaf_folder = (
            "" if not ft_manager.enabled else f"replica_{ft_manager.replica_id}"
        )

        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as memory_profiler,
            maybe_semi_sync_training(
                job_config.fault_tolerance,
                ft_manager=ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    self.model_args.n_layers
                    if hasattr(self.model_args, "n_layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    self.train_spec.fragment_fn
                    if hasattr(self.train_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # Run validation if validator is available
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")


if __name__ == "__main__":
    main(FTTrainer)
