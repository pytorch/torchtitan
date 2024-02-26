import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

from pippy.PipelineSchedule import PipelineStageV2Impl as PipelineStageV2ImplOrig
from pippy.PipelineSchedule import PipelineScheduleGPipe as PipelineScheduleGPipeOrig
from pippy.PipelineSchedule import create_buffers

logger = logging.getLogger(__name__)

class PipelineStageV2Impl(PipelineStageV2ImplOrig):

    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        rank: int,
        world_size: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
        output_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
        label_arg: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):

        super().__init__(
            module,
            stage_id,
            num_stages,
            rank,
            world_size,
            device,
            input_args,
            output_args,
        )

        self.label = None
        if label_arg is not None:
            self.label = create_buffers(label_arg, device)[0]
            assert isinstance(self.label, torch.Tensor), f"label must be a tensor, but got {type(label)}"
        self.loss_fn = loss_fn


    """
    Fixes (relative to upstream)
    - only set requires grad on later stages (input batch is int64 which can't have requires_grad)
    - work around passing first stage's inputs to autograd.grad
    - return loss from forward
    - hacky implement compute_loss

    - try sending labels from first stage through all intermediate stages to last stage
    - seems harder to get a direct first-to-last stage send/recv to work?
    - need to allocate recv buffers on non-0 stages for labels, need labels shapes
    - should stop passing real data in on non-0 ranks
    """

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.irecv, inp, self.prev_stage) for inp in self.inputs
        ] + [
            dist.P2POp(dist.irecv, self.label, self.prev_stage)
        ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        assert (
            len(self.outputs) != 0
        ), "forward() must be called before get_fwd_send_ops"
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.isend, out, self.next_stage) for out in self.outputs
        ] + [
            dist.P2POp(dist.isend, self.label, self.next_stage)
        ]


    def compute_loss(self):
        if self.outputs is None:
            raise RuntimeError("forward() must be called before compute_loss()")
        return self.loss_fn(self.outputs[0], self.label)

    def forward(self, args: Union[torch.Tensor, List[torch.tensor]], *, label) -> Any:
        if self.is_first_stage:
            # we always expect to unpack an iterable of inputs, so if its a single tensor, wrap it in a list
            if isinstance(args, torch.Tensor):
                args = [args]
            self.inputs = args
            assert isinstance(label, torch.Tensor), f"label must be a tensor, but got {type(label)}"
            self.label = label

        logger.info(
            f"[{self.rank} FORWARD {self.stage_id} {[inp.shape for inp in self.inputs]}"
        )

        # this is needed when we access the gradients for this in backward()
        # TODO: requires_grad should not be set, it should depend on input (https://github.com/pytorch/PiPPy/issues/945)
        if not self.is_first_stage:
            for tensor in self.inputs:
                tensor.requires_grad = True
                tensor.retain_grad()

        # perform forward pass on module
        outputs = self.module(*self.inputs)
        self.outputs = self.check_and_format_outputs(outputs)

        outputs_or_loss = self.compute_loss() if self.is_last_stage else outputs

        # we store a ref to the input/output pair for this forward to be later used by the corresponding backward
        self.inputs_outputs.append((self.inputs, outputs_or_loss))

        return outputs_or_loss

    def backward(self) -> None:
        logger.info(f"[{self.rank} BACKWARD {self.stage_id}]")
        inputs, outputs = self.inputs_outputs.popleft()

        # Compute gradients
        torch.autograd.backward(
            tensors=outputs,
            grad_tensors=None if self.is_last_stage else self.outputs_grad,
        )
        self.inputs_grad = [x.grad for x in inputs]


class PipelineScheduleGPipe(PipelineScheduleGPipeOrig):
    def step(self, microbatches, labels):
        mb_loss = []
        for i, (mb, label) in enumerate(zip(microbatches, labels)):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                outputs_or_loss = self._stage.forward(mb, label=label)
                if self._stage.is_last_stage:
                    mb_loss.append(outputs_or_loss.clone().detach())

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

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

        if self._stage.is_last_stage:
            return mb_loss
