# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is not a production training recipe. It is just dummy incomplete code
to demonstrate observability APIs.

Toy RL with Monarch actors: RollouterActor produces completions,
RewardActor scores them, TrainerActor trains on them. Controller
orchestrates the loop.

Run (requires 4 GPUs):
    python -m torchtitan.experiments.observability.toy_rl

Do NOT use torch.distributed.run / torchrun — the controller is a single
process that spawns GPU workers via Monarch's this_host().spawn_procs().
"""

import asyncio
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.observability.toy_spmd import (
    BATCH_SIZE,
    DP_SIZE,
    SEQ_LEN,
    setup_data,
    ToyTrainer,
    VOCAB_SIZE,
)
from torchtitan.observability import (
    EventType,
    init_observability,
    record_span,
    set_step,
)
from torchtitan.observability.analysis import generate_gantt_trace
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

NUM_STEPS = 6
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_rl")


# ---------------------------------------------------------------------------
# RolloutOutput
# ---------------------------------------------------------------------------


@dataclass
class RolloutOutput:
    """One prompt+completion pair from a rollout.

    Token fields and training tensors are used for training.
    Text fields are for logging and human inspection only — in a real
    pipeline they would be populated by tokenizer.decode().
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    prompt_text: str
    completion_text: str
    reward: float | None = None
    # Training tensors (one sample, not batched)
    tokens: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None

    def to_logging_dict(self) -> dict:
        """Convert to a dict for logging. Only text + reward."""
        d = {"prompt": self.prompt_text, "completion": self.completion_text}
        if self.reward is not None:
            d["reward"] = self.reward
        return d


def rollouts_to_train_batch(
    rollouts: list[RolloutOutput],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack individual rollout tensors into a training batch.

    Returns:
        (tokens, labels, loss_mask), each of shape (batch_size, seq_len).
    """
    return (
        torch.stack([r.tokens for r in rollouts]),
        torch.stack([r.labels for r in rollouts]),
        torch.stack([r.loss_mask for r in rollouts]),
    )


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


class RollouterActor(Actor):
    """Dummy rollouter that returns fixed data as if it were generated."""

    @endpoint
    async def setup(self):
        rank = current_rank().rank
        init_logger()
        init_observability(source="rollouter", output_dir=OUTPUT_DIR, rank=rank)
        dataset = setup_data(batch_size=DP_SIZE * BATCH_SIZE)
        self.dataset = dataset

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller."""
        set_step(step)

    @endpoint
    async def do_rollouts(self, prompts: torch.Tensor) -> list[RolloutOutput]:
        """Produce rollouts. Each carries one sample's tensors and dummy text.

        This is a dummy — not a real generation loop. In a real pipeline,
        the model would generate completions and the tokenizer would
        decode them into text.
        """
        with record_span("rollouter_time/generate_s"):
            rollouts = [
                RolloutOutput(
                    prompt_tokens=self.dataset.tokens[i].tolist(),
                    completion_tokens=self.dataset.tokens[i].tolist(),
                    prompt_text=f"What is {i}+{i}?",
                    completion_text=f"The answer is {i + i}.",
                    tokens=self.dataset.tokens[i],
                    labels=self.dataset.labels[i],
                    loss_mask=self.dataset.loss_mask[i],
                )
                for i in range(len(self.dataset.tokens))
            ]
        return rollouts


class TrainerActor(Actor):
    @endpoint
    async def setup(self):
        rank = current_rank().rank
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(self.device)
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            os.environ.setdefault("MASTER_PORT", "29500")
            world_size = int(os.environ.get("WORLD_SIZE", 4))
            torch.distributed.init_process_group(
                backend="nccl", rank=rank, world_size=world_size
            )
        init_logger()
        init_observability(source="trainer", output_dir=OUTPUT_DIR, rank=rank)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=DP_SIZE,
            cp=1,
            tp=world_size // DP_SIZE,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()
        self.dp_rank = parallel_dims.get_mesh("fsdp").get_local_rank()
        self.trainer = ToyTrainer(self.device, parallel_dims, OUTPUT_DIR)

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller. Sets step on trainer."""
        self.trainer.step = step
        self.trainer.metrics_processor.set_step(step)

    @endpoint
    async def train_step(self, tokens, labels, loss_mask):
        """Train one step on generated completions."""
        # Slice this DP rank's shard from the full batch.
        start = self.dp_rank * BATCH_SIZE
        end = start + BATCH_SIZE
        tokens = tokens[start:end].to(self.device)
        labels = labels[start:end].to(self.device)
        loss_mask = loss_mask[start:end].to(self.device)

        # Train step (loss is logged inside via logger.info)
        self.trainer.train_step(tokens, labels, loss_mask)

    @endpoint
    async def teardown(self):
        self.trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class RewardActor(Actor):
    """Dummy reward actor. Scores are not used by the trainer — this actor
    exists to demonstrate multi-actor observability patterns."""

    @endpoint
    async def setup(self):
        rank = current_rank().rank
        init_logger()
        init_observability(source="reward", output_dir=OUTPUT_DIR, rank=rank)

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller."""
        set_step(step)

    @endpoint
    async def score(self, rollouts: list[RolloutOutput]) -> list[RolloutOutput]:
        """Score rollouts. Fills in reward field and returns them."""
        with record_span("reward_time/scoring_s"):
            for rollout in rollouts:
                rollout.reward = 1.0  # dummy constant reward
        return rollouts


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


async def main():
    t0 = time.time()
    logger.info(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_logger()
    init_observability(source="controller", output_dir=OUTPUT_DIR, rank=0)

    # ---- Setup ----
    host = this_host()

    rollouter_mesh = host.spawn_procs(per_host={"procs": 1}, name="rollouter")
    rollouter = rollouter_mesh.spawn("rollouter", RollouterActor)
    await rollouter.setup.call()

    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()

    actors = [rollouter, trainer, reward_actor]
    logger.info("Actors spawned.")

    # Dummy prompts for the rollouter.
    prompts = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # ---- Training loop ----
    async def run_training():
        for step in range(1, NUM_STEPS + 1):
            set_step(step)
            for actor in actors:
                await actor.set_step.call(step)

            with record_span("rl_time/rollout_s", EventType.RL_ROLLOUT):
                result = await rollouter.do_rollouts.call(prompts)
                rollouts = next(iter(result.values()))

            with record_span("rl_time/scoring_s", EventType.RL_GRADING):
                result = await reward_actor.score.call(rollouts)
                rollouts = next(iter(result.values()))

            with record_span("rl_time/rollouts_to_train_batch_s"):
                tokens, labels, loss_mask = rollouts_to_train_batch(rollouts)

            with record_span("rl_time/training_s", EventType.FWD_BWD):
                await trainer.train_step.call(tokens, labels, loss_mask)

            rewards = [r.reward for r in rollouts]
            reward_mean = sum(rewards) / len(rewards)
            logger.info(f"step: {step}  reward_mean: {reward_mean:.5f}")

    await run_training()

    # ---- Cleanup ----
    await trainer.teardown.call()

    sys_logs = os.path.join(OUTPUT_DIR, "system_logs")
    trace_path = os.path.join(OUTPUT_DIR, "analysis", "system_metrics_gantt.json")
    generate_gantt_trace(sys_logs, trace_path)

    logger.info(f"Done in {time.time() - t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
