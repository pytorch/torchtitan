# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion loop recipes: the DistMoE MXFP8 capture with a fusion plan lifted and kernels substituted.

FUSION_PLAN: comma-separated lift passes from fusion_regions (this directory).
LOOP_ITER: output subdirectory (iter1, iter2, ...).
Kernels come from KERNEL_CACHE via substitute_with_kernels_pass (PR #5838); a miss runs
the lifted reference, so numerics must stay bitwise identical either way.
"""

import os

import example_recipe

import torch
from fusion_finder_passes import enable_fusion_finder

from torchtitan.experiments.graph_trainer.registry import register_graph_pass
from torchtitan.experiments.graph_trainer.substitute_kernels import (
    substitute_with_kernels_pass,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# KERNEL_CACHE overrides the cache, e.g. a copy with reference stubs to test the hit path.
KERNEL_CACHE = os.environ.get("KERNEL_CACHE", os.path.join(HERE, "kernel_cache"))


@register_graph_pass
def substitute_kernels(gm, example_inputs=None):
    return substitute_with_kernels_pass(gm, example_inputs, cache_dir=KERNEL_CACHE)


def lifted():
    import fusion_regions  # noqa: F401  (registers the lift passes)

    plan = [p for p in os.environ["FUSION_PLAN"].split(",") if p]
    config = enable_fusion_finder(
        example_recipe.dsv3_671b_base(), os.path.join(HERE, os.environ["LOOP_ITER"])
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
    """Save the real arguments of the first call of every lifted fusion_* submodule."""
    out = os.path.join(HERE, "real_inputs")
    os.makedirs(out, exist_ok=True)
    for target in {n.target for n in gm.graph.find_nodes(op="call_module")}:
        if target.startswith("fusion_"):
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
        example_recipe.dsv3_671b_base(), os.path.join(HERE, "capture_inputs")
    )
    config.compile.extra_passes = [*plan, "substitute_kernels", "capture_region_inputs"]
    config.training.steps = 1
    config.profiler.enable_profiling = False
    return config
