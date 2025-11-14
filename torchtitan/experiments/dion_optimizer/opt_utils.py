# This code is from:
# https://github.com/microsoft/dion/blob/main/optimizers/opt_utils.py


# @article{ahn2025dion,
#  title={Dion: Distributed Orthonormalized Updates},
#  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
#  journal={arXiv preprint: 2504.05295},
#  year={2025}
# }

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
from typing import Generator, List, Optional, Union

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor


def to_local(tensor: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
    """
    Convert a single DTensor or list of DTensors to local tensors.
    This is a no-op for regular tensors.
    """
    if isinstance(tensor, Tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]


def dtensor_from_local(
    tensor: Union[Tensor, List[Tensor]], ref: Tensor
) -> Union[DTensor, List[DTensor]]:
    """
    Convert a single local Tensor or list of local Tensors to DTensor.
    The reference tensor's device mesh and placements are used to create the DTensor.
    if the reference tensor is not a DTensor, we return the input unmodified.
    """
    if not isinstance(ref, DTensor):
        assert isinstance(ref, Tensor)
        return tensor

    device_mesh = ref.device_mesh
    placements = ref.placements

    # If we have a single tensor
    if isinstance(tensor, Tensor):
        assert not isinstance(tensor, DTensor)
        return DTensor.from_local(
            tensor, device_mesh=device_mesh, placements=placements
        )

    # We have a list of tensors
    assert not isinstance(tensor[0], DTensor)
    return [
        DTensor.from_local(t, device_mesh=device_mesh, placements=placements)
        for t in tensor
    ]


def create_param_batches(
    params: List[Tensor], batch_size: int
) -> Generator[List[Tensor], None, None]:
    """
    Batch parameters into groups of size `batch_size`.
    Tensors in each batch will have identical shape, sharding, and dtype.
    """
    # Group parameters by shape, sharding, and dtype
    groups = defaultdict(list)
    for p in params:
        sharding = p.placements if isinstance(p, DTensor) else None
        groups[(p.shape, sharding, p.dtype)].append(p)

    # Create batches from grouped parameters
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            yield batch


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """
    Insert dummy tensors so the batch has exactly `batch_size` elements.
    """
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.empty_like(batch[0]))
    return batch


class AsyncTask:
    """
    AsyncTask wraps a Python generator to run until the next yield statement.
    This is used to allow other tasks to run while waiting for distributed operations.
    """

    def __init__(self, generator: Generator[None, None, None]):
        self._generator = generator
        self.run()  # Start running the generator

    def run(self) -> bool:
        # Run the next step of the async task.
        # Returns True if the task is still running and False if completed.
        try:
            next(self._generator)
            return True
        except StopIteration:
            pass
        return False


class AsyncRuntime:
    """
    Event loop for running multiple async tasks concurrently.
    """

    def __init__(
        self, task_gen: Generator["AsyncTask", None, None], max_concurrent_tasks: int
    ):
        # Initialize runtime with a generator that produces AsyncTask objects
        if max_concurrent_tasks <= 0:
            raise ValueError(f"{max_concurrent_tasks=} cannot be <= 0")
        self._task_gen = task_gen
        self._max_concurrent_tasks = max_concurrent_tasks

    def _get_next_task(self) -> Optional["AsyncTask"]:
        try:
            task = next(self._task_gen)
            return task
        except StopIteration:
            return None

    def run(self):
        # Run the event loop until all tasks are completed
        have_new_tasks = True
        previous_tasks: List["AsyncTask"] = []

        while have_new_tasks or previous_tasks:
            # See if we can add another task
            running_tasks = []
            if have_new_tasks and len(previous_tasks) < self._max_concurrent_tasks:
                new_task = self._get_next_task()
                if new_task is not None:
                    # Add new task to the queue
                    running_tasks.append(new_task)
                else:
                    # No more tasks left
                    have_new_tasks = False

            # Run all previous tasks for one step
            for task in previous_tasks:
                still_running = task.run()
                if still_running:
                    running_tasks.append(task)

            # Update task list for next iteration
            previous_tasks = running_tasks
