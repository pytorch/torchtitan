# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agentic compiler loop for graph_trainer_gpt_oss_120b_mxfp8_64gpu on one fake-PG GPU.

Run from the repo root with the fusion_finder scripts and this directory on PYTHONPATH:

    S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts
    D=torchtitan/experiments/graph_trainer/agentic_compiler/gpt_oss_120b_mxfp8_64gpu
    env FUSION_PLAN=$(cat $D/plans/plan_iter3.txt) LIFT_OPTIONAL=1 RUN=full_final \\
        PYTHONPATH=$S:$D:. NGPU=64 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 \\
        PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train --module loop_recipe --config full_lifted --comm-backend fake

Configs (outputs go to $D/$RUN):
- ``full`` / ``full_det``: the benchmark recipe as shipped (5 microbatches per step), 10 steps,
  TensorBoard; ``full_det`` adds seed 42 and deterministic algorithms (the bitwise gate).
- ``capture``: one microbatch per step under ``enable_fusion_finder`` (profiled step 10).
- ``lifted``: ``capture``'s setup plus FUSION_PLAN (comma-separated lift passes from
  ``fusion_regions``) and substitute_kernels from KERNEL_CACHE.
- ``full_lifted``: ``full_det`` plus the same passes, then ``unsubstituted_to_call_function``.
  With LIFT_OPTIONAL=1 a lift pass whose region has no instance in the graph is skipped.
- ``lifted_inputs``: one unprofiled step of ``lifted`` that also saves the first real arguments of
  every lifted submodule to real_inputs/<submodule>.pt (for kernel verification).
"""

import os

import torch
from fusion_finder_passes import enable_fusion_finder

from torchtitan.experiments.graph_trainer.registry import register_graph_pass
from torchtitan.experiments.graph_trainer.substitute_kernels import (
    substitute_with_kernels_pass,
)
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    graph_trainer_gpt_oss_120b_mxfp8_64gpu,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = os.path.join(HERE, "capture")
KERNEL_CACHE = os.environ.get("KERNEL_CACHE", os.path.join(HERE, "kernel_cache"))


@register_graph_pass
def substitute_kernels(gm, example_inputs=None):
    return substitute_with_kernels_pass(gm, example_inputs, cache_dir=KERNEL_CACHE)


def _run_dir(default):
    return os.path.join(HERE, os.environ.get("RUN", default))


def _plan():
    import fusion_regions  # noqa: F401  (registers the lift passes)

    plan = [p for p in os.environ.get("FUSION_PLAN", "").split(",") if p]
    if os.environ.get("LIFT_OPTIONAL"):
        return [_optional(p) for p in plan]
    return plan


def _optional(name):
    """Register ``optional_<name>``: ``name``, but a region with no instance is skipped."""
    from torchtitan.experiments.graph_trainer.registry import GRAPH_PASS_REGISTRY

    lift = GRAPH_PASS_REGISTRY[name]

    def optional(gm, example_inputs=None):
        try:
            return lift(gm, example_inputs)
        except ValueError as e:
            if "no instance to lift" not in str(e):
                raise
            print(f"[loop_recipe] skipping {name}: {e}", flush=True)
            return gm

    optional.__name__ = f"optional_{name}"
    register_graph_pass(optional)
    return optional.__name__


def full():
    config = graph_trainer_gpt_oss_120b_mxfp8_64gpu()
    config.training.steps = 10
    out = _run_dir("full")
    config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(out, "tb")
    config.dump_folder = out
    return config


def full_det():
    config = full()
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def _one_microbatch():
    """The recipe with one microbatch per step, so a profiled step is one forward and backward."""
    config = graph_trainer_gpt_oss_120b_mxfp8_64gpu()
    config.training.num_tokens_per_train_step = -1
    return config


def capture():
    return enable_fusion_finder(_one_microbatch(), CAPTURE)


def lifted():
    plan = _plan()
    config = enable_fusion_finder(_one_microbatch(), _run_dir("lifted"))
    config.compile.extra_passes = [*plan, "substitute_kernels"]
    return config


def _as_function(name, module):
    def fn(*args):
        return module(*args)

    fn.__name__ = fn.__qualname__ = name
    return fn


@register_graph_pass
def unsubstituted_to_call_function(gm, example_inputs=None):
    """Call the lifted fusion_* submodules that substitute_kernels left in place as functions.

    The gradient-accumulation graph builder deep-copies graphs and runs DCE on the copies,
    which have no owning module, so a remaining call_module node fails its purity check.
    A call_function node keeps the call site's own meta (inlining the submodule's graph
    would copy the first site's meta["val"] to every site, and q and k RoPE share one).
    """
    fns = {}
    for node in list(gm.graph.find_nodes(op="call_module")):
        if not node.target.startswith("fusion_"):
            continue
        if node.target not in fns:
            fns[node.target] = _as_function(node.target, gm.get_submodule(node.target))
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(fns[node.target], node.args, node.kwargs)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
    for target in fns:
        gm.delete_submodule(target)
    gm.recompile()
    return gm


def full_lifted():
    plan = _plan()
    config = full_det()
    config.compile.extra_passes = [
        *plan,
        "substitute_kernels",
        "unsubstituted_to_call_function",
    ]
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
        if target.startswith("fusion_"):
            path = os.path.join(out, f"{target}.pt")
            gm.add_submodule(target, _SaveFirstCall(gm.get_submodule(target), path))
    return gm


def lifted_inputs():
    config = lifted()
    config.compile.extra_passes = [
        *config.compile.extra_passes,
        "capture_region_inputs",
    ]
    config.training.steps = 1
    config.profiler.enable_profiling = False
    return config
