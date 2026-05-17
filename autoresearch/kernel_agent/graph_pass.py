"""Graph pass that replaces matched subgraphs with fused custom ops.

Registers a pass pipeline "fused_kernels" that extends the default pipeline
with a fused kernel replacement pass. The pass loads generated kernels from
autoresearch/kernel_agent/generated/ and dispatches to the best backend
(triton, torch.compile, or eager) based on benchmark results.

This module must be imported before the trainer queries the pass pipeline
registry. The simplest way is to add to the run command:

  NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b \\
      FUSED_KERNELS=1 ./run_train.sh \\
      --compile.mode aot_fx_trace \\
      --compile.pass_pipeline fused_kernels

Or import explicitly in Python:
  import autoresearch.kernel_agent.graph_pass  # registers "fused_kernels" pipeline

Environment variables:
  FUSED_KERNEL_BACKEND       — default backend for all ops (triton/compile/eager)
  FUSED_KERNEL_OVERRIDES     — per-op overrides: "op_name:backend,op_name:backend"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from torchtitan.experiments.graph_trainer.passes import construct_default_graph_passes
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline

logger = logging.getLogger(__name__)

_GENERATED_DIR = Path(__file__).parent / "generated"


@register_pass_pipeline("fused_kernels")
def fused_kernels_pipeline(traced_result, config):
    """Default passes + fused kernel replacement pass."""
    passes = construct_default_graph_passes(traced_result, config)

    from autoresearch.kernel_agent.fused_ops import FusedOpRegistry

    registry = FusedOpRegistry.from_generated(_GENERATED_DIR)

    if not registry.ops:
        logger.warning("No fused ops found in %s, skipping fused kernel pass", _GENERATED_DIR)
        return passes

    # Apply backend selection from environment
    default_backend = os.environ.get("FUSED_KERNEL_BACKEND")
    if default_backend:
        registry.set_default_backend(default_backend)
        logger.info(f"Fused kernels: default backend={default_backend}")

    overrides = os.environ.get("FUSED_KERNEL_OVERRIDES", "")
    for item in overrides.split(","):
        item = item.strip()
        if ":" in item:
            op_name, backend = item.split(":", 1)
            try:
                registry.set_backend(op_name, backend)
                logger.info(f"Fused kernels: {op_name} → {backend}")
            except ValueError as e:
                logger.warning(f"Fused kernels: {e}")

    # Log what's loaded
    for name, op in sorted(registry.ops.items()):
        active = op.active_backend or op.best_backend
        backends = list(op.implementations.keys())
        logger.info(f"  {name}: backends={backends}, active={active}")

    fused_pass = registry.make_graph_pass()
    passes.append(fused_pass)

    return passes
