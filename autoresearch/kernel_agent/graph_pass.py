"""Graph pass that replaces matched subgraphs with fused custom ops.

Registers a pass pipeline "fused_kernels" that extends the default pipeline
with a fused kernel replacement pass. The pass loads generated kernels from
autoresearch/kernel_agent/generated/ and dispatches to the best backend
(triton, torch.compile, or eager) based on benchmark.json results.

This is the legacy pipeline approach — prefer using the config-based
integration: --compile.fused_kernel_dir <path>
"""

from __future__ import annotations

import logging
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

    for name, op in sorted(registry.ops.items()):
        backends = list(op.implementations.keys())
        logger.info(f"  {name}: backends={backends}, best={op.best_backend}")

    fused_pass = registry.make_graph_pass()
    passes.append(fused_pass)

    return passes
