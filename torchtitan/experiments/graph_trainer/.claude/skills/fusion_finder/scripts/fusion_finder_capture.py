# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture one GraphTrainer training step for the fusion finder.

A recipe calls ``configure_capture(config, out_dir)``. The step then runs the
``fusion_finder`` pass pipeline: the default GraphTrainer passes, followed by

- ``dump_graph_pass``, which writes ``<out_dir>/graph.jsonl.gz``: one JSON record
  per node of the graph that runs, including its line of FX code, the storage it
  writes or aliases, its ``match_key`` (what instances of a region share), and
  whether it exists before the final Inductor passes (where
  ``compile.extra_passes`` run);
- ``custom_codegen_pass``, which runs every node under
  ``_RecordFunctionFast("<node>: <op> (<file>:<line>)")``, so that each kernel in
  the profiler trace can be attributed to the node that launched it.
"""

import functools
import gzip
import json
import operator
import os
import re
from typing import Any

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree

from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.passes import (
    construct_default_graph_passes,
    final_inductor_compile_passes,
)
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline

_PROFILE_STEP = 10
_FRAME_RE = re.compile(r'File "(.+)", line (\d+), in (\S+)')
# A node's line in FX code: <name>[: "<dtype>[<shape>][<stride>]<device>"] = ...
_CODE_RE = re.compile(r'\s*(\w+)(?:: "([^"]*)")? = (.*)')


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


def encode(x: Any) -> Any:
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
        return [encode(v) for v in x]
    if isinstance(x, dict):
        return {str(k): encode(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    return str(x)


def target_name(node: fx.Node) -> str:
    t = node.target
    if isinstance(t, (str, torch._ops.OpOverload)):
        return str(t)
    if isinstance(t, torch._ops.HigherOrderOperator):
        return t.name()
    return f"{getattr(t, '__module__', '')}.{getattr(t, '__qualname__', t)}"


def node_phase(node: fx.Node) -> str:
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


def _code_lines(gm: fx.GraphModule) -> dict[str, str]:
    """Each node's line in the FX code of ``gm``'s top-level graph, with shapes and
    strides, without the trailing ``;  <name> = None`` frees. The type annotation is
    single-quoted so that the line needs no escaping in JSON."""
    src = gm.graph.python_code(
        "self", verbose=True, include_stride=True, include_device=True
    ).src
    lines = {}
    for line in src.splitlines():
        m = _CODE_RE.match(line)
        if m:
            name, annotation, rhs = m.groups()
            typed = f"{name}: '{annotation}'" if annotation else name
            lines[name] = f"{typed} = {rhs.split(';  ')[0]}"
    return lines


def storage_effects(node: fx.Node) -> tuple[list[str], list[str]]:
    """Names of the inputs whose storage ``node`` writes, and of the inputs whose
    storage its output aliases, from the op schema (``Tensor(a!)``, ``Tensor(a)``).

    A ``getitem`` aliases its input.
    """
    if node.target is operator.getitem:
        source = node.args[0]
        return [], [source.name] if isinstance(source, fx.Node) else []
    if not isinstance(node.target, torch._ops.OpOverload):
        return [], []
    schema = node.target._schema
    writes, aliased = [], []
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None:
            continue
        if not arg.kwarg_only and i < len(node.args):
            value = node.args[i]
        elif arg.name in node.kwargs:
            value = node.kwargs[arg.name]
        else:
            continue
        names = [x.name for x in pytree.tree_leaves(value) if isinstance(x, fx.Node)]
        aliased += names
        if arg.alias_info.is_write:
            writes += names
    if not any(r.alias_info is not None for r in schema.returns):
        aliased = []
    return writes, aliased


# Size arguments typed int[] instead of SymInt[] in their schema.
# TODO: make _fused_rms_norm's normalized_shape SymInt[] upstream, as rms_norm's is.
_INT_LIST_SIZE_ARGS = {
    ("aten::_fused_rms_norm", "normalized_shape"),
    ("aten::_fused_rms_norm_backward", "normalized_shape"),
}


def _argument_kinds(node: fx.Node) -> dict[int | str, str]:
    """``size`` or ``scalar`` per argument position and name of ``node``'s op, for
    the arguments whose values may differ between instances of a region."""
    if not isinstance(node.target, torch._ops.OpOverload):
        return {}
    schema = node.target._schema
    kinds: dict[int | str, str] = {}
    for i, arg in enumerate(schema.arguments):
        t = arg.real_type
        while isinstance(t, (torch._C.OptionalType, torch._C.ListType)):
            t = t.getElementType()
        if (
            isinstance(t, torch._C.SymIntType)
            or (schema.name, arg.name) in _INT_LIST_SIZE_ARGS
        ):
            kind = "size"
        elif isinstance(t, (torch._C.TensorType, torch._C.NumberType)):
            kind = "scalar"
        else:
            continue
        kinds[arg.name] = kind
        if not arg.kwarg_only:
            kinds[i] = kind
    return kinds


def _match_value(value: Any, kind: str | None) -> Any:
    if isinstance(value, (list, tuple)):
        return [_match_value(v, kind) for v in value]
    if isinstance(value, bool):
        return value
    if kind == "size" and isinstance(value, (int, torch.SymInt, fx.Node)):
        return "*"
    if kind == "scalar" and isinstance(
        value, (int, float, torch.SymInt, torch.SymFloat)
    ):
        return "*"
    if isinstance(value, torch.device):
        return value.type
    return encode(value)


def match_key(node: fx.Node) -> list:
    """``[target, args, kwargs]`` of ``node`` as regions compare it, with tensor
    inputs as ``{"node": name}`` and ``"*"`` for the values that may differ between
    instances of a region: sizes (arguments the op schema types ``SymInt``) and
    numbers passed as a ``Tensor`` or ``Scalar``. Devices are their type."""
    kinds = _argument_kinds(node)
    args = [_match_value(v, kinds.get(i)) for i, v in enumerate(node.args)]
    kwargs = {k: _match_value(v, kinds.get(k)) for k, v in node.kwargs.items()}
    return [target_name(node), args, kwargs]


def _note_node_names(
    gm: fx.GraphModule, example_inputs: Any = None, *, names: set[str]
) -> fx.GraphModule:
    names.update(node.name for node in gm.graph.nodes)
    return gm


def dump_graph_pass(
    gm: fx.GraphModule,
    example_inputs: Any = None,
    *,
    path: str,
    before_inductor: set[str],
) -> fx.GraphModule:
    """Write one JSON record per node of ``gm`` to ``path`` (rank 0 only)."""
    if torch.distributed.get_rank() != 0:
        return gm
    code = _code_lines(gm)
    with gzip.open(path, "wt") as f:
        for node in gm.graph.nodes:
            writes, aliases = storage_effects(node)
            record = {
                "name": node.name,
                "op": node.op,
                "target": target_name(node),
                "args": encode(node.args),
                "kwargs": encode(node.kwargs),
                "match_key": match_key(node),
                "val": encode(node.meta.get("val")),
                "phase": node_phase(node),
                "module_fqn": node.meta.get("custom", {}).get("module_fqn", ""),
                "src": _source_line(node.meta.get("stack_trace")),
                "code": code.get(node.name, ""),
                "writes": writes,
                "aliases": aliases,
                "before_inductor": node.name in before_inductor,
            }
            f.write(json.dumps(record) + "\n")
    return gm


@register_pass_pipeline("fusion_finder")
def fusion_finder_pipeline(traced, trainer_config, *, parallelism_context):
    out_dir = trainer_config.dump_folder
    passes = construct_default_graph_passes(
        traced, trainer_config, parallelism_context=parallelism_context
    )
    # Note the nodes of the graph that compile.extra_passes see, before Inductor
    # compiles the tagged regions into new nodes.
    num_inductor_passes = len(
        final_inductor_compile_passes(
            trainer_config.compile,
            use_cuda_graph=not trainer_config.training.disable_accelerator_graphs,
        )
    )
    before_inductor: set[str] = set()
    passes.insert(
        len(passes) - num_inductor_passes,
        functools.partial(_note_node_names, names=before_inductor),
    )
    return [
        *passes,
        functools.partial(
            dump_graph_pass,
            path=os.path.join(out_dir, "graph.jsonl.gz"),
            before_inductor=before_inductor,
        ),
        functools.partial(
            custom_codegen_pass, codegen_dir=os.path.join(out_dir, "codegen")
        ),
    ]
