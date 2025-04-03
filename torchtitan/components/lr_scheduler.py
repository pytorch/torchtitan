# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import math
from typing import Any, Callable, Iterator

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

__all__ = [
    "LRSchedulersContainer",
    "build_lr_schedulers",
]


class LRSchedulersContainer(Stateful):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Note**
    Users who want to customize the lr_scheduler behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same
    signature as ``torch.optim.lr_scheduler.LRScheduler`` class: ``step()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes all the lr schedulers are the same. There is no easy way to support
    resharding for multiple different LRSchedulers because LRScheduler.state_dict() is not
    resharding friendly. Therefore, the limitation is used to allow TorchTitan to support
    lr scheduler resharding.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    schedulers: list[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert (
            len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because
        # the state_dict is the same for all. See the limitations section in the
        # docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``training.steps`` and ``lr_scheduler.warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))


def build_lr_schedulers(
    optimizers: OptimizersContainer, job_config: JobConfig
) -> LRSchedulersContainer:
    """Create a LRSchedulerContainer for the given optimizers and job config.

    This function creates a ``LRSchedulersContainer`` for the given optimizers.
    ``job_config`` should define the correct lr scheduler parameters.

    **Note**
    Users who want to customize the lr scheduler behavior can create their own
    ``LRSchedulersContainer`` subclass and ``build_lr_scheduler``. Passing the
    customized ``build_lr_schedulers`` to ``TrainSpec`` will create the customized
    ``LRSchedulersContainer``.


    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the
            lr_schedulers.
    """
    training_steps = job_config.training.steps
    warmup_steps = int(job_config.lr_scheduler.warmup_steps)
    if job_config.lr_scheduler.decay_ratio is not None:
        decay_steps = round(training_steps * job_config.lr_scheduler.decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            logger.warning(
                f"Warmup ({warmup_steps}) + decay ({decay_steps}) steps exceed "
                f"total training steps ({training_steps}). "
                f"Adjusting decay steps to {training_steps - warmup_steps}."
            )
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps
    stable_steps = training_steps - warmup_steps - decay_steps
    lr_decay_type = job_config.lr_scheduler.decay_type
    lr_min = job_config.lr_scheduler.lr_min

    def linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_decay_type: str,
        lr_min: float,
    ):
        """
        Computes linear warmup followed by stable learning rate for a while,
        then some type of decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor `curr_adjustment` ranging from 1 to 0
        to adjust the learning rate to create the desired schedule.

        We offer three types of learning rate decay schedules:
        1. `linear`: decays linearly from 1 to 0 over the decay period.
        2. `sqrt`: decays as 1 minus the square root of the decay progress.
        3. `cosine`: follows a cosine curve, decaying according to the values of the half-period of the cosine function.

        If `lr_min` is specified, the decay range is scaled from 1 to `lr_min`
        to ensure the learning rate does not drop below this minimum value.
        """
        warmup_stable_steps = warmup_steps + stable_steps
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        else:
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            progress = float(current_step - warmup_stable_steps) / (decay_steps + 1)

            if lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif lr_decay_type == "cosine":
                curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            curr_adjustment = lr_min + (1 - lr_min) * curr_adjustment
        return curr_adjustment

    lr_lambda = functools.partial(
        linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_decay_type,
        lr_min=lr_min,
    )
    return LRSchedulersContainer(optimizers, lr_lambda)
