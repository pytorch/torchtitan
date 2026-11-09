# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion loop recipes: Qwen3 235B-A22B MXFP8 Dist-MoE on one fake-PG GB300.

The base is ``graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu`` (FSDP 256 x EP 32,
8K-token microbatches) with one microbatch per step and FA4 varlen attention, as in the
Qwen3 30B-A3B loop. Run with ``NGPU=256 LOCAL_RANK=0 --comm-backend fake``.

FUSION_FINDER_OUT: baseline capture directory (``baseline``).
FUSION_PLAN: comma-separated lift passes from fusion_regions (this directory).
LOOP_ITER: output subdirectory of this directory (iter1, iter2, ...).
KERNEL_CACHE: substitute-kernels cache (default: kernel_cache in this directory).

``baseline`` and ``lifted`` profile step 10 of 10 with the fusion_finder pipeline (CUDA graphs
off); ``timed_baseline`` and ``timed`` run 20 unprofiled steps of the recipe as is (default
pipeline, CUDA graphs on). All use debug.deterministic and seed 42, and log TensorBoard for
the bitwise check.
"""

import os

import torch
from fusion_finder_passes import enable_fusion_finder

from torchtitan.experiments.graph_trainer.registry import register_graph_pass
from torchtitan.experiments.graph_trainer.substitute_kernels import (
    substitute_with_kernels_pass,
)
from torchtitan.models.common.attention.attention import VarlenInnerAttention

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu,
)

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_CACHE = os.environ.get("KERNEL_CACHE", os.path.join(HERE, "kernel_cache"))


def qwen3_235b_base():
    """The benchmark recipe with one microbatch per step and FA4 varlen attention.

    cuDNN varlen backward is nondeterministic even with ``debug.deterministic``; FA4
    honors it, so the bitwise gate is meaningful.
    """
    config = graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu()
    config.training.num_tokens_per_train_step = -1
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    return config


def baseline():
    out = os.environ.get("FUSION_FINDER_OUT", os.path.join(HERE, "baseline"))
    return enable_fusion_finder(qwen3_235b_base(), out)


@register_graph_pass
def substitute_kernels(gm, example_inputs=None):
    return substitute_with_kernels_pass(gm, example_inputs, cache_dir=KERNEL_CACHE)


def lifted():
    import fusion_regions  # noqa: F401  (registers the lift passes)

    plan = [p for p in os.environ["FUSION_PLAN"].split(",") if p]
    config = enable_fusion_finder(
        qwen3_235b_base(), os.path.join(HERE, os.environ["LOOP_ITER"])
    )
    config.compile.extra_passes = [*plan, "substitute_kernels"]
    return config


class _SaveFirstCall(torch.nn.Module):
    """Runs ``inner``; on the first call also saves its arguments (on CPU) to ``path``."""

    def __init__(self, inner, path):
        super().__init__()
        self.inner, self.path, self.done = inner, path, False

    def forward(self, *args):
        if not self.done:
            torch.save(
                [a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args],
                self.path + ".tmp",
            )
            os.replace(self.path + ".tmp", self.path)
            self.done = True
        return self.inner(*args)


@register_graph_pass
def capture_region_inputs(gm, example_inputs=None):
    """Save the real arguments of the first call of every lifted fusion_* submodule."""
    out = os.path.join(HERE, "real_inputs")
    os.makedirs(out, exist_ok=True)
    for target in {n.target for n in gm.graph.find_nodes(op="call_module")}:
        if target.startswith("fusion_") and not os.path.exists(
            os.path.join(out, f"{target}.pt")
        ):
            gm.add_submodule(
                target,
                _SaveFirstCall(
                    gm.get_submodule(target), os.path.join(out, f"{target}.pt")
                ),
            )
    return gm


def capture():
    """The plan's regions lifted, saving each region's first real inputs; one step."""
    import fusion_regions  # noqa: F401

    plan = [p for p in os.environ["FUSION_PLAN"].split(",") if p]
    config = enable_fusion_finder(
        qwen3_235b_base(), os.path.join(HERE, "capture_inputs")
    )
    config.compile.extra_passes = [*plan, "substitute_kernels", "capture_region_inputs"]
    config.training.steps = 1
    config.profiler.enable_profiling = False
    return config


def _timed(config, out):
    """20 unprofiled steps with the recipe's CUDA graphs on (det, seed 42); TensorBoard for the bitwise check."""
    from torchtitan.observability.metrics import TensorBoardLogger

    config.training.steps = 20
    config.debug.seed = 42
    config.debug.deterministic = True
    config.profiler.enable_profiling = False
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(HERE, out, "tb")
    config.dump_folder = os.path.join(HERE, out)
    return config


def timed_baseline():
    """The base recipe, unlifted: the end-to-end reference for ``timed``."""
    return _timed(qwen3_235b_base(), "timed_baseline")


def timed():
    """The plan's regions lifted and substituted from KERNEL_CACHE, timed like ``timed_baseline``."""
    import fusion_regions  # noqa: F401

    config = _timed(qwen3_235b_base(), os.environ["LOOP_ITER"])
    config.compile.extra_passes = [
        *[p for p in os.environ["FUSION_PLAN"].split(",") if p],
        "substitute_kernels",
    ]
    return config
