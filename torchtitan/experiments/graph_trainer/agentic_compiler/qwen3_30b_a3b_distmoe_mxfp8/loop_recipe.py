# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion loop recipes: Qwen3 30B-A3B MXFP8 Dist-MoE on one fake-PG GB300.

The base is ``graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu`` (FSDP 8 x EP 8,
32K-token microbatches) with one microbatch per step, so the profiled step is one
forward and backward, and FA4 instead of cuDNN attention (see ``qwen3_30b_base``).
Run with ``NGPU=8 LOCAL_RANK=0 --comm-backend fake``.

FUSION_FINDER_OUT: baseline capture directory (``baseline``).
FUSION_PLAN: comma-separated lift passes from fusion_regions (this directory).
LOOP_ITER: output subdirectory of this directory (iter1, iter2, ...).
Kernels come from KERNEL_CACHE via substitute_with_kernels_pass (PR #5838); a miss runs
the lifted reference, so numerics must stay bitwise identical either way.
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
    graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# KERNEL_CACHE overrides the cache, e.g. a copy with reference stubs to test the hit path.
KERNEL_CACHE = os.environ.get("KERNEL_CACHE", os.path.join(HERE, "kernel_cache"))


def qwen3_30b_base():
    """The benchmark recipe with one microbatch per step and FA4 varlen attention.

    Round-robin routing is exactly balanced even with every EP peer aliasing this
    rank under the fake PG, so the recipe's 1.5x Dist-MoE receive capacity holds.

    The recipe's varlen attention prefers cuDNN, whose backward is nondeterministic
    even with ``debug.deterministic`` (two identical runs differ in step-1 grad_norm),
    so the bitwise gate would be meaningless. FA4 honors deterministic mode; the
    lifted regions are glue outside attention, so their problems are unchanged.
    """
    config = graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu()
    config.training.num_tokens_per_train_step = -1
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    return config


def baseline():
    out = os.environ.get(
        "FUSION_FINDER_OUT", "outputs/fusion_finder/qwen3_30b_a3b_distmoe_mxfp8"
    )
    return enable_fusion_finder(qwen3_30b_base(), out)


@register_graph_pass
def substitute_kernels(gm, example_inputs=None):
    return substitute_with_kernels_pass(gm, example_inputs, cache_dir=KERNEL_CACHE)


def lifted():
    import fusion_regions  # noqa: F401  (registers the lift passes)

    plan = [p for p in os.environ["FUSION_PLAN"].split(",") if p]
    config = enable_fusion_finder(
        qwen3_30b_base(), os.path.join(HERE, os.environ["LOOP_ITER"])
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
            # Write then rename: kernel writers may be reading the previous file.
            torch.save(
                [a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args],
                self.path + ".tmp",
            )
            os.replace(self.path + ".tmp", self.path)
            self.done = True
        return self.inner(*args)


@register_graph_pass
def capture_region_inputs(gm, example_inputs=None):
    """Save the real arguments of the first call of every lifted fusion_* submodule.

    Existing files are kept, so the inputs behind earlier kernel evidence stay the same.
    """
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
    """The plan's regions lifted, saving each unsubstituted region's first real inputs.

    Substitution runs first so cache misses also write their reference and specialization.
    """
    import fusion_regions  # noqa: F401

    plan = [p for p in os.environ["FUSION_PLAN"].split(",") if p]
    config = enable_fusion_finder(
        qwen3_30b_base(), os.path.join(HERE, "capture_inputs")
    )
    config.compile.extra_passes = [*plan, "substitute_kernels", "capture_region_inputs"]
    config.training.steps = 1
    config.profiler.enable_profiling = False
    return config
