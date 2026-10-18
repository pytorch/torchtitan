# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop on an actor framework (Monarch by default; see ``Controller.Config.actor_system``).

This demonstrates:
1. Distributed actor architecture with Generator (vLLM) and Trainer (TorchTitan)
   running on separate GPU meshes
2. Model state dict transfer across meshes via TorchStore: the trainer publishes its
   model state dict and the generator pulls it into its own parallelism layout,
   with direct GPU-to-GPU RDMA transfer when available
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 -m torchtitan.rl.train \
    --module my_rl_configs --config rl_grpo_qwen3_0_6b_varlen

Set ``hf_assets_path`` in ``my_rl_configs.py`` to the model checkpoint path.
"""

# Apply process env defaults (e.g. the CUDA allocator config) before anything
# imports torch; spawned procs re-apply them in their bootstrap.
from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import asyncio
import logging
import os
import sys

from torchtitan.config import ConfigLoader
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.system import ActorSystem
from torchtitan.rl.controller.controller import Controller


logger = logging.getLogger(__name__)


async def main() -> ActorSystem | None:
    """Run RL training; return the actor system for the caller to shut down."""
    init_logger()
    config = ConfigLoader().load()
    assert isinstance(config, Controller.Config)
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer: Controller = config.build()
    try:
        await rl_trainer.setup_async()
        await rl_trainer.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
        raise
    finally:
        await rl_trainer.close()
    return rl_trainer.actor_system


if __name__ == "__main__":
    actor_system = asyncio.run(main())
    # The run finished and every actor is closed. Shut down the actor framework's
    # client (for Monarch, normally an atexit hook), flush logs, then exit without
    # interpreter finalization: before CPython 3.14, finalization ends Monarch's
    # daemon threads with pthread_exit while they are inside PyO3 calls, and PyO3
    # swallowing that forced unwind aborts the process (exit 134). Errors raised
    # above still take the normal exit path.
    if actor_system is not None:
        actor_system.shutdown()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
