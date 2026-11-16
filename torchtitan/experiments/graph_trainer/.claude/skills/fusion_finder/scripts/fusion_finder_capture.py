# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture one GraphTrainer training step for the fusion finder.

A recipe calls ``configure_capture(config, out_dir)``. The step then runs the
``fusion_finder`` pass pipeline: the default GraphTrainer passes, followed by

- ``dump_graph_pass``, which writes ``<out_dir>/graph.jsonl.gz``: one JSON record
  per node of the graph that runs;
- ``custom_codegen_pass``, which runs every node under
  ``_RecordFunctionFast("<node>: <op> (<file>:<line>)")``, so that each kernel in
  the profiler trace can be attributed to the node that launched it.
"""

import functools
import gzip
import json
import os
import re
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.passes import construct_default_graph_passes
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline

_PROFILE_STEP = 10
_FRAME_RE = re.compile(r'File "(.+)", line (\d+), in (\S+)')


def configure_capture(config: Any, out_dir: str) -> Any:
    """Profile step 10 of ``config`` (a ``GraphTrainer.Config``) into ``out_dir``.

    The trace goes to ``<out_dir>/profile/iteration_10/rank0_trace.json`` and the
    graph to ``<out_dir>/graph.jsonl.gz``.
    """
    # The pass pipeline only applies to a one-microbatch step without PP.
    if config.training.num_tokens_per_train_step >= 0:
        raise ValueError(
            "the fusion finder profiles one microbatch per step: "
            "set training.num_tokens_per_train_step = -1"
        )
    if config.parallelism.pipeline_parallel_degree > 1:
        raise ValueError("the fusion finder does not support pipeline parallelism")
    # After full Inductor compilation the graph is one compiled call.
    if config.compile.inductor_compilation != "regional":
        raise ValueError(
            "the fusion finder requires compile.inductor_compilation='regional'"
        )
    config.dump_folder = os.path.abspath(out_dir)
    config.compile.pass_pipeline = "fusion_finder"
    # Accelerator graph replay would hide the per-node labels.
    config.training.disable_accelerator_graphs = True
    config.training.steps = _PROFILE_STEP
    config.profiler.enable_profiling = True
    config.profiler.profile_freq = _PROFILE_STEP
    config.profiler.save_traces_folder = "profile"
    # Kernels written for the regions found must match this step bit for bit, so
    # profile the deterministic code paths.
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def _encode(x: Any) -> Any:
    if isinstance(x, fx.Node):
        return {"node": x.name}
    if isinstance(x, torch.Tensor):
        return {
            "t": "tensor",
            "shape": [s if isinstance(s, int) else str(s) for s in x.shape],
            "stride": [s if isinstance(s, int) else str(s) for s in x.stride()],
            "dtype": str(x.dtype).removeprefix("torch."),
            "device": x.device.type,
        }
    if isinstance(x, (list, tuple)):
        return [_encode(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _encode(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    return str(x)


def _target_name(node: fx.Node) -> str:
    t = node.target
    if isinstance(t, (str, torch._ops.OpOverload)):
        return str(t)
    if isinstance(t, torch._ops.HigherOrderOperator):
        return t.name()
    return f"{getattr(t, '__module__', '')}.{getattr(t, '__qualname__', t)}"


def _phase(node: fx.Node) -> str:
    # selective_activation_remat_pass names its forward copies <name>_recomputed[_<n>].
    if "_recomputed" in node.name:
        return "recompute"
    return "backward" if node.meta.get("autograd_backward") else "forward"


def _source_line(stack_trace: str | None) -> str:
    """The innermost frame outside PyTorch: ``<path>:<line> in <function>``."""
    for path, line, func in reversed(_FRAME_RE.findall(stack_trace or "")):
        if "/torch/" not in path and "/site-packages/" not in path:
            return f"{os.path.relpath(path)}:{line} in {func}"
    return ""


def dump_graph_pass(
    gm: fx.GraphModule, example_inputs: Any = None, *, path: str
) -> fx.GraphModule:
    """Write one JSON record per node of ``gm`` to ``path`` (rank 0 only)."""
    if torch.distributed.get_rank() != 0:
        return gm
    with gzip.open(path, "wt") as f:
        for node in gm.graph.nodes:
            record = {
                "name": node.name,
                "op": node.op,
                "target": _target_name(node),
                "args": _encode(node.args),
                "kwargs": _encode(node.kwargs),
                "val": _encode(node.meta.get("val")),
                "phase": _phase(node),
                "module_fqn": node.meta.get("custom", {}).get("module_fqn", ""),
                "src": _source_line(node.meta.get("stack_trace")),
            }
            f.write(json.dumps(record) + "\n")
    return gm


@register_pass_pipeline("fusion_finder")
def fusion_finder_pipeline(traced, trainer_config, *, parallelism_context):
    out_dir = trainer_config.dump_folder
    return [
        *construct_default_graph_passes(
            traced, trainer_config, parallelism_context=parallelism_context
        ),
        functools.partial(
            dump_graph_pass, path=os.path.join(out_dir, "graph.jsonl.gz")
        ),
        functools.partial(
            custom_codegen_pass, codegen_dir=os.path.join(out_dir, "codegen")
        ),
    ]
