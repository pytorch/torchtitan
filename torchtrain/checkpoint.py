# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import enum
import os
import re
import time
from contextlib import nullcontext
from multiprocessing import get_context
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._state_dict_utils import (
    _create_cpu_state_dict,
    _offload_state_dict_to_cpu,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchtrain.logging_utils import init_logger, rank0_log


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


class Terminate:
    pass


class SaveDone:
    pass


def checkpoint_mp(recv, send):
    init_logger()
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 1)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    while True:
        rank0_log("Checkpoint background process is done")
        send.put(SaveDone())
        rank0_log("Wait for the new state_dict.")
        obj = recv.get()
        rank0_log("Received the new state_dict.")
        if isinstance(obj, Terminate):
            rank0_log("Terminating the checkpoint backgroun process.")
            return
        assert isinstance(obj, tuple)
        begin = time.monotonic()
        state, checkpoint_id = obj
        dcp.save(state, checkpoint_id=checkpoint_id)
        rank0_log(
            "Finish saving the checkpoint in the backgroun process. "
            f"{time.monotonic() - begin} seconds"
        )


class CheckpointManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        states: Dict[str, Any],
        folder: str,
        interval_type: IntervalType,
        interval: int,
        enable_mp: bool = False,
        overlap_with_training: bool = False,
    ) -> None:
        self.folder = folder
        self.states = states
        self.states.update(
            {
                "model": ModelWrapper(model),
                "optimizer": OptimizerWrapper(model, optimizer),
            }
        )
        self.interval_type = interval_type
        self.interval = interval
        self.begin = 0
        self.work = None
        self.pg = dist.new_group(backend="gloo")
        self.doit = None
        self.staging = False

        if enable_mp:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
        else:
            self.mp = None

        self.cpu_offload_state_dict = None
        self.staging_stream = torch.cuda.Stream()
        self.overlap_with_training = overlap_with_training

    def __del__(self):
        if self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self) -> None:
        self.begin = time.monotonic()

    def create_checkpoint_id(self, step: int) -> str:
        return os.path.join(self.folder, f"step-{step}")

    def save(self, curr_step: int, force: bool = False) -> None:
        if not self.folder:
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

        begin = time.monotonic()
        rank0_log(f"Saving a checkpoint in step {curr_step}. {begin}")
        checkpoint_id = self.create_checkpoint_id(curr_step)
        if self.mp:
            # Make sure the checkpoint process is done
            rank0_log(
                f"Waiting for the background process to finish, {time.monotonic()}."
            )
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()

            rank0_log(f"Expanding the stateful, {time.monotonic()}")
            state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
            rank0_log(f"Prepare the memory, {time.monotonic()}")
            if self.cpu_offload_state_dict is None:
                self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict)

            rank0_log(f"Staging the state_dict, {time.monotonic()}")
            ctx = (
                torch.cuda.stream(self.staging_stream)
                if self.overlap_with_training
                else nullcontext()
            )
            with ctx:
                self.cpu_offload_state_dict = _offload_state_dict_to_cpu(
                    state_dict,
                    cpu_offload_state_dict=self.cpu_offload_state_dict,
                    cpu_offload_sync=False,
                    type_check=False,
                )
                self.staging = True

            rank0_log(
                f"Sending the state dict to the background process, {time.monotonic()}."
            )
            self.mp_queue_send.put((state_dict, checkpoint_id))
            rank0_log(f"Finish the async_save, {time.monotonic()}.")
        else:
            dcp.save(self.states, checkpoint_id=checkpoint_id)
        self.reset()
        rank0_log(
            f"Finish saving the checkpoint in step {curr_step}. "
            f"{time.monotonic() - begin} seconds. "
        )

    def load(self, step: int = -1) -> bool:
        if not self.folder:
            return False
        if not os.path.isdir(self.folder):
            return False
        if step != -1 and not os.path.isdir(self.create_checkpoint_id(step)):
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

        rank0_log("Loading a checkpoint.")
        begin = time.monotonic()
        dcp.load(
            self.states,
            checkpoint_id=self.create_checkpoint_id(step),
        )
        rank0_log(f"Finish loading a checkpoint. {time.monotonic() - begin} seconds.")
        return True

    def wait_staging(self) -> None:
        if self.staging and self.overlap_with_training:
            begin = time.monotonic()
            self.staging_stream.synchronize()
            rank0_log(
                f"Finish synchronizing the stagning stream, {time.monotonic() - begin}"
            )
        self.staging = False
