# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import os
import re
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class IntervalType(enum.Enum):
    SECONDS = enum.auto()
    STEPS = enum.auto()


class ModelWrapper:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> None:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(self.model, state_dict)


class OptimizerWrapper:
    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer) -> None:
        self.model = model
        self.optim = optim

    def state_dict(self) -> None:
        return get_optimizer_state_dict(self.model, self.optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(self.model, self.optim, optim_state_dict=state_dict)


class CheckpointManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        states: Dict[str, Any],
        job_config: JobConfig,
    ) -> None:
        ckpt_config = job_config.checkpoint
        self.enable_checkpoint = ckpt_config.enable_checkpoint

        if self.enable_checkpoint:
            self.states = states
            self.states.update(
                {
                    "model": ModelWrapper(model),
                    "optimizer": OptimizerWrapper(model, optimizer),
                }
            )

            self.folder = os.path.join(job_config.job.dump_folder, ckpt_config.folder)
            self.interval_type = (
                IntervalType.SECONDS
                if ckpt_config.interval_type == "seconds"
                else IntervalType.STEPS
            )
            self.interval = ckpt_config.interval
            self.model_weights_only = ckpt_config.model_weights_only
            self.export_dtype = DTYPE_MAP[ckpt_config.export_dtype]

            logger.info(
                f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
            )

            self.begin = 0
            self.work = None
            self.pg = dist.new_group(backend="gloo")
            self.doit = None

    def reset(self) -> None:
        self.begin = time.monotonic()

    def _create_checkpoint_id(self, step: int) -> str:
        return os.path.join(self.folder, f"step-{step}")

    def save(self, curr_step: int, force: bool = False) -> None:
        """
        force = True will force the checkpoint to be saved, even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps.
        """
        if not self.enable_checkpoint:
            return

        if not force:
            if self.interval_type == IntervalType.STEPS and not (
                curr_step % self.interval == 0
            ):
                return
            if self.interval_type == IntervalType.SECONDS:
                doit = (time.monotonic() - self.begin) >= self.interval
                self.doit = torch.tensor(int(doit))
                if self.work is None:
                    self.work = dist.all_reduce(self.doit, group=self.pg, async_op=True)
                    return
                elif curr_step % 5 == 4:
                    self.work.wait()
                    self.work = None
                    doit = self.doit.item()
                    self.doit = None
                    if doit == 0:
                        return
                else:
                    return

        if self.work:
            self.work.wait()
            self.work = None
            self.doit = None

        # We only consider saving weights only at the end of the training.
        # So this won't affect preemption and training resume.
        # We also only allow dtype conversion when we are checkpoint model weights only
        # and the current dtype is not the same as the export dtype at the end of the training.
        if force and self.model_weights_only:
            # We update self.states to keep the model only.
            # After this update, self.states = {'tok_embeddings.weight':...,''layers.0.attention.wq.weight': ...}.
            self.states = self.states["model"].state_dict()

            # For now, we will manually pop the freqs_cis buffer, as we made this permanent
            # temporarily and we don't want to include it in the exported state_dict.
            # Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L348
            self.states.pop("freqs_cis")

            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }
            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} at step {curr_step}"
            )

        else:
            logger.info(f"Saving a full checkpoint at step {curr_step}")

        begin = time.monotonic()
        dcp.save(self.states, checkpoint_id=self._create_checkpoint_id(curr_step))
        self.reset()
        logger.info(
            f"Finished saving the checkpoint in {time.monotonic() - begin:.2f} seconds"
        )

    def load(self, step: int = -1) -> bool:
        if not self.enable_checkpoint:
            return False
        if not os.path.isdir(self.folder):
            return False
        if step != -1 and not os.path.isdir(self._create_checkpoint_id(step)):
            return False

        if step == -1:
            step_counts = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    step_counts.append(int(match.group(1)))
            if not step_counts:
                return False
            step = max(step_counts)

        # We won't have optimizer states to load, if we are loading a seed checkpoint
        states = {"model": self.states["model"]} if step == 0 else self.states
        logger.info(f"Loading the checkpoint at step {step}")
        begin = time.monotonic()
        dcp.load(
            states,
            checkpoint_id=self._create_checkpoint_id(step),
        )
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds"
        )
        return True
