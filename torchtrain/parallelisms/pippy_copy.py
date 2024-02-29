import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function
from torch.distributed.distributed_c10d import get_global_rank

from pippy.PipelineSchedule import PipelineStageV2Impl as PipelineStageV2ImplOrig
from pippy.PipelineSchedule import PipelineScheduleGPipe as PipelineScheduleGPipeOrig
from pippy.PipelineSchedule import create_buffers, create_metadata_tensor, PipelineStage, extract_metadata_from_tensor

logger = logging.getLogger(__name__)

def get_stage_shapes(
    models: List[nn.Module],
    stage_ids: List[int],
    num_stages: int,
    rank: int,
    world_size: int,
    group,
    device: torch.device,
    microbatch: Optional[Union[torch.tensor, List[torch.tensor]]] = None,
):
    """
    Performs a dry run through all the pipeline stages (a rank can have multiple pipeline stages in the case of
    virtual pipelining) and returns the shape of the inputs and outputs of the module.
    Only the first stage must pass in a microbatch.

    Each rank must call get_stage_shapes or the program will hang.

    Args:
        models: The chunks assigned to this rank. Rhe length should be 1 for any
                non-interleaved schedules and >1 for any interleaved schedules.
        stage_ids: The id of the stages assigned to this rank.
        num_stages: Total number of stages.
        rank: Rank of the current process.
        world_size: Number of processes participating in the pipeline.
        device: Device where the tensors are allocated.

    Returns a dictionary containing the following keys:
        "inputs": Shape of the inputs to the module
        "outputs": Shape of the outputs of the module
    """

    stage_id_to_shapes: Dict[int, Dict[str, torch.Size]] = {}
    for stage_id, model in zip(stage_ids, models, strict=True):
        input_shape_metadata_tensor = create_metadata_tensor(device=device)
        # TODO: Assumes prev_stage == rank - 1 and next_stage == rank + 1
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        shapes = {}

        # first stage doesn't receive anything and uses a microbatch
        if stage_id == 0:
            if microbatch is None:
                raise RuntimeError("Microbatch is required for first stage")
            example_fwd_inputs = microbatch
            if isinstance(example_fwd_inputs, torch.Tensor):
                example_fwd_inputs = [example_fwd_inputs]
        else:
            # other stages must receive shape information
            # TODO: send/recv should take a group, rather than use the default group
            # dist.recv(input_shape_metadata_tensor, prev_rank, group=group)
            group.recv([input_shape_metadata_tensor], prev_rank, tag=0).wait()
            metadata = extract_metadata_from_tensor(input_shape_metadata_tensor)
            example_fwd_inputs = [
                torch.empty(shape_list, device=device)
                for shape_list in metadata
            ]
        shapes["inputs"] = [fwd_input.shape for fwd_input in example_fwd_inputs]

        # perform forward
        # TODO: if forward fails raise a more descriptive error explaining which stage failed
        fwd_outputs = model(*example_fwd_inputs)
        fwd_outputs = create_buffers(fwd_outputs, device)
        shapes["outputs"] = [fwd_output.shape for fwd_output in fwd_outputs]

        # send shape dims
        if stage_id != num_stages - 1:
            output_shape_metadata_tensor = create_metadata_tensor(
                fwd_outputs, device=device
            )
            # dist.send(output_shape_metadata_tensor, next_rank, group=group)
            group.send([output_shape_metadata_tensor], next_rank, tag=0).wait()
        stage_id_to_shapes[stage_id] = shapes
    print(stage_id_to_shapes)
    return stage_id_to_shapes


def validate_stage_shapes(pipeline_stages: List[PipelineStage], group):
    """
    Check that the buffer shapes match between stages was expected by performing an all_gather between
    all stages. Assumes that buffers have been initialized already such that get_fwd_recv_ops() and
    get_fwd_send_ops() return valid lists of p2p ops.
    """
    virtual_pipeline_size = len(pipeline_stages)
    all_inputs = []
    all_outputs = []
    # perform all gathers between all stages
    for virtual_id, stage in enumerate(pipeline_stages):
        world_size = stage.world_size
        stage_id = stage.stage_id
        rank = stage.rank

        # TODO: once we pass in pg to stage, check the pg rank is same as stage rank
        if rank != (pg_rank := dist.get_rank()):
            raise ValueError(
                f"Rank {rank} is not equal to process group rank {pg_rank}"
            )

        if (num_stages := stage.num_stages) % world_size != 0:
            raise ValueError(
                f"Number of stages ({num_stages}) must be a multiple of the world_size ({world_size})"
            )

        # all gather each ranks inputs
        tensor_list = [
            create_metadata_tensor(device=stage.device)
            for _ in range(stage.world_size)
        ]
        expected_inputs = [op.tensor for op in stage.get_fwd_recv_ops()]
        stage_input = create_metadata_tensor(
            expected_inputs, device=stage.device
        )
        dist.all_gather(tensor_list, stage_input, group=group)
        stage_input_shapes = [
            extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        # all gather each ranks outputs
        tensor_list = [
            create_metadata_tensor(device=stage.device)
            for _ in range(stage.world_size)
        ]
        expected_outputs = [op.tensor for op in stage.get_fwd_send_ops()]
        stage_output = create_metadata_tensor(
            expected_outputs, device=stage.device
        )
        dist.all_gather(tensor_list, stage_output, group=group)
        stage_output_shapes = [
            extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        logger.debug(
            f"""
            Rank: {pg_rank}
            Stage id: {stage_id}
            Stage num stages: {stage.num_stages}
            Stage rank: {rank}
            Stage world size: {world_size}
            Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} input shapes: {stage_input_shapes}
            Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} output shapes: {stage_output_shapes}
        """
        )

        all_inputs.extend(stage_input_shapes)
        all_outputs.extend(stage_output_shapes)

    # log only rank 0's view, they will all be equivalent
    if pg_rank == 0:
        logger.info(
            f"""
            all stage inputs: {all_inputs}
            all stage outputs: {all_outputs}
        """
        )

    # Check if the output for stage 0 matches the input at stage 1, and so forth
    for i in range(virtual_pipeline_size * world_size - 1):
        if (out := all_outputs[i]) != (inp := all_inputs[i + 1]):
            raise ValueError(
                f"Stage_id {stage_id} output shape {out} at does not match stage_id {i + 1} input shape {inp}."
            )



class PipelineStageV2Impl(PipelineStageV2ImplOrig):

    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
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


        self.prev_stage = get_global_rank(group, (rank - 1) % world_size)
        self.next_stage = get_global_rank(group, (rank + 1) % world_size)

        logger.info(
            f"""
            updated!!!
            finished pipeline stage init, {self.stage_id=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=}, {self.prev_stage=}, {self.next_stage=},
            {[inp.shape for inp in self.inputs]},
            {[output.shape for output in self.outputs]}
            """
        )

        self.label = None
        if label_arg is not None:
            self.label = create_buffers(label_arg, device)[0]
            assert isinstance(self.label, torch.Tensor), f"label must be a tensor, but got {type(label)}"
        self.loss_fn = loss_fn
        # TODO(whc)
        self.group = group


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
        dist.recv(self.inputs[0], self.prev_stage)
        dist.recv(self.label, self.prev_stage)
        return []
        # return [
        #     dist.P2POp(dist.irecv, inp, self.prev_stage) for inp in self.inputs
        # ]
        #  + [
        #     dist.P2POp(dist.irecv, self.label, self.prev_stage)
        # ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        assert (
            len(self.outputs) != 0
        ), "forward() must be called before get_fwd_send_ops"
        assert self.label[0,0] > 0 and self.label[0,0] < 65000, "bad label before send"
        if self.is_last_stage:
            return []

            # WTF. coalesced mode seems to give garbage, individual sends seem ok
        dist.send(self.inputs[0], self.next_stage)
        dist.send(self.label, self.next_stage)
        return []
        # return [
        #     dist.P2POp(dist.isend, out, self.next_stage) for out in self.outputs
        # ]
        # + [
        #     dist.P2POp(dist.isend, self.label, self.next_stage)
        # ]


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
            self.label.copy_(label)

            assert self.label[0,0] > 0 and self.label[0,0] < 65000, "bad label first stage forward"

        if not self.is_first_stage:
            # logger.info(f"stage got inputs {self.inputs[0][:10]}, dtype {self.inputs[0].dtype}")
            logger.info(f"stage got label {self.label[0][:10]}, dtype {self.label[0].dtype}")

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

        # logger.info(f"stage sent outputs {self.outputs[0][:10]}")

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
