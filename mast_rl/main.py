"""Python entry point for Search-R1 RL training on MAST.

A single process; Monarch spawns trainer/generator actors across the host's
GPUs (see ``torchtitan/experiments/rl/train.py``). The retrieval server is
started separately by ``run.sh`` before this runs.

This differs from the older ``fb/mast_rl/main.py`` (which imported
``torchtitan.experiments.rl.grpo``): the current OSS RL experiment's entry
point is ``torchtitan.experiments.rl.train:main``.
"""

import asyncio

from torchtitan.experiments.rl.train import main as rl_main


if __name__ == "__main__":
    asyncio.run(rl_main())
