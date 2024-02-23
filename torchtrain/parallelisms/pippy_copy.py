from pippy.PipelineSchedule import PipelineStage, PipelineSchedule


# copied from hhuang's refactor PR https://github.com/pytorch/PiPPy/blob/916f26092f8cac52383040c19a47feb6fd473b88/pippy/PipelineSchedule.py
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

logger = logging.getLogger(__name__)

class PipelineStageV2Impl(PipelineStage):
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        rank: int,
        world_size: int,
        inputs_meta: List[torch.tensor],
        outputs_meta: Optional[List[torch.tensor]] = None,
    ):
        super().__init__()
        self.module = module
        self.rank = rank
        self.stage_id = stage_id
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
        self.num_stages = num_stages

        if outputs_meta is None:
            # since building PP stage after fsdp wrapping, model isn't on meta anymore, use real inputs for shape infr
            outputs_meta = module.forward(inputs_meta)
        self.fwd_inputs: List[torch.tensor] = self.create_buffers(inputs_meta)
        self.fwd_outputs = None
        self.fwd_output_grads: List[torch.tensor] = self.create_buffers(
            outputs_meta
        )
        self.fwd_outputs_for_backward: Deque[
            Tuple[torch.tensor, torch.tensor]
        ] = deque()
        logger.debug(
            f"""
            {[fwd_input.shape for fwd_input in self.fwd_inputs if isinstance(fwd_input, torch.Tensor)]},
            {[fwd_output_grad.shape for fwd_output_grad in self.fwd_output_grads]}
            """
        )

        self.prev_stage = (rank - 1) % world_size
        self.next_stage = (rank + 1) % world_size

        self.fwd_recv_queue = None
        self.bwd_recv_queue = None

        self.requests: List[dist.P2POp] = []
        logger.debug(
            f"""
            finished pipeline stage init, {self.stage_id=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=},
            """
        )

    def to(self, *args, **kwargs):
        """
        Move the module to a new device or data type, including the buffers.
        """
        super().to(*args, **kwargs)

        # find the device of the underlying module and move the buffers to it if they are meta
        device = next(self.module.parameters()).device

        for i, fwd_input in enumerate(self.fwd_inputs):
            if fwd_input.is_meta:
                self.fwd_inputs[i] = torch.empty_like(fwd_input, device=device)
            self.fwd_inputs[i] = self.fwd_inputs[i].to(*args, **kwargs)
        for i, fwd_output_grad in enumerate(self.fwd_output_grads):
            if fwd_output_grad.is_meta:
                self.fwd_output_grads[i] = torch.empty_like(
                    fwd_output_grad, device=device
                )
            self.fwd_output_grads[i] = self.fwd_output_grads[i].to(
                *args, **kwargs
            )

    def create_buffers(
        self, inputs_meta: List[torch.tensor]
    ) -> List[torch.Tensor]:
        """
        Creates buffers for a given input on a specified device.
        This function takes as input a meta tensor or a list of meta tensors
        and returns a flattened list of empty tensors of the same shape.
        """
        if isinstance(inputs_meta, torch.Tensor):
            return [inputs_meta]
        elif isinstance(inputs_meta, (list, tuple)):
            return [
                item
                for sublist in [self.create_buffers(inp) for inp in inputs_meta]
                for item in sublist
            ]
        raise ValueError(
            f"Unsupported input type {type(inputs_meta)} cannot create buffers"
        )

    def init_p2p_neighbors(self):
        """
        Set up p2p communitors between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        ops = []
        recv_tensor = torch.zeros(1, device="cuda")
        send_tensor = torch.ones(1, device="cuda")
        # forward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_stage))

        # backward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_stage))

        return True

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.irecv, fwd_input, self.prev_stage)
            for fwd_input in self.fwd_inputs
        ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        assert (
            self.fwd_outputs is not None
        ), "forward() must be called before get_fwd_send_ops"
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.isend, fwd_output, self.next_stage)
            for fwd_output in self.fwd_outputs
        ]

    def forward(self, args: List[torch.tensor]) -> torch.tensor:
        logger.debug(f"[{self.rank} FORWARD {self.stage_id}")
        if self.is_first_stage:
            self.fwd_inputs = args

        # this is needed when we access the gradients for this in backward()
        if not self.is_first_stage:
            # TODO(whc) - did we need this for the first stage for some reason? hope not bc first stage inputs are
            # int64 and autograd doesn't like that
            for tensor in self.fwd_inputs:
                tensor.requires_grad = True
                tensor.retain_grad()

        # perform forward pass on module
        self.fwd_outputs = self.module(self.fwd_inputs)

        output_for_backward = (
            self.compute_loss() if self.is_last_stage else self.fwd_outputs
        )

        # we store a ref to the input/output pair for this forward to be later used by the corresponding backward
        self.fwd_outputs_for_backward.append(
            (self.fwd_inputs, output_for_backward)
        )

        return self.fwd_outputs

    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.irecv, output_grad, self.next_stage)
            for output_grad in self.fwd_output_grads
        ]

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        for fwd_input in self.fwd_inputs:
            logger.debug(f"{fwd_input.grad=}")
            assert fwd_input.grad is not None, "grad must be valid"
        return [
            dist.P2POp(dist.isend, fwd_input.grad, self.prev_stage)
            for fwd_input in self.fwd_inputs
        ]

    def backward(self):
        logger.debug(f"[{self.rank} BACKWARD {self.stage_id}]")

        self.fwd_inputs, fwd_outputs_or_loss = self.fwd_outputs_for_backward.popleft()

        # Compute gradients
        torch.autograd.backward(
            tensors=fwd_outputs_or_loss,
            grad_tensors=None if self.is_last_stage else self.fwd_output_grads,
        )

        return self.fwd_inputs

    def compute_loss(self):
        if self.fwd_outputs is None:
            raise RuntimeError("forward() must be called before compute_loss()")
        # TODO: use a real loss function passed in
        return self.fwd_outputs[0].mean()


class PipelineScheduleGPipe(PipelineSchedule):
    def __init__(self, stage: PipelineStage):
        self._stage = stage

    def step(self, microbatches):
        for i, mb in enumerate(microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.forward(mb)

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

                logger.info(
                    f"{self._stage.stage_id} forward mb {i} finished, microbatch: {[inp.shape for inp in mb]}"
                )

        for i, _ in enumerate(microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.backward()

                ops = self._stage.get_bwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

            logger.info(f"{self._stage.stage_id} backward mb {i} finished")
