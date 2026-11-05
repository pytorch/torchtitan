# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture instrumentation for the fusion region finder.

Registers two GraphTrainer pass pipelines:

* ``fusion_finder``: the default pipeline plus
    - ``dump_graph_pass(tag="pre_inductor")`` before the final Inductor passes,
    - ``dump_graph_pass(tag="final")`` after all default passes (the graph that runs),
    - ``custom_codegen_pass`` last, so each FX node runs under
      ``_RecordFunctionFast("node_name: op (file:line)")`` while the profiler is on.
* ``fusion_finder_recompute_index_put``: the same plus
  ``recompute_fresh_buffer_index_put_pass`` after memory-policy tagging (see its
  docstring; needed when ``memory_policy="full"`` runs out of memory on MoE models).

A recipe imports this module and calls ``enable_fusion_finder(config, out_dir)``.
Nothing under ``torchtitan/`` is modified.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import json
import logging
import os
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.passes import (
    _get_pass_name,
    construct_default_graph_passes,
)
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline
from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter
from torchtitan.observability.metrics import TensorBoardLogger

logger = logging.getLogger(__name__)

_OUT_ENV = "FUSION_FINDER_OUT"


def enable_fusion_finder(
    config: Any,
    out_dir: str,
    *,
    pipeline: str = "fusion_finder",
    profile_step: int = 10,
    profiler_warmup: int = 3,
) -> Any:
    """Configure ``config`` (a GraphTrainer.Config) for one fusion-finder capture."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.environ[_OUT_ENV] = out_dir
    config.compile.pass_pipeline = pipeline
    config.training.steps = profile_step
    config.debug.seed = 42
    config.debug.deterministic = True
    config.profiler.enable_profiling = True
    config.profiler.enable_memory_snapshot = False
    config.profiler.save_traces_folder = out_dir
    config.profiler.profile_freq = profile_step
    config.profiler.profiler_warmup = profiler_warmup
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    # Graph replay hides the per-node _RecordFunctionFast labels that attribute kernels.
    config.training.disable_accelerator_graphs = True
    # Full-precision loss and grad_norm, to check lifted runs bitwise against this one.
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(out_dir, "tb")
    if list(config.model.local_compile.regions):
        raise ValueError("GraphTrainer requires model.local_compile.regions=[]")
    setup = {
        "pass_pipeline": pipeline,
        "memory_policy": config.compile.memory_policy,
        "inductor_compilation": config.compile.inductor_compilation,
        "parallelism": {
            k: getattr(config.parallelism, k)
            for k in (
                "data_parallel_shard_degree",
                "data_parallel_replicate_degree",
                "tensor_parallel_degree",
                "context_parallel_degree",
                "pipeline_parallel_degree",
                "expert_parallel_degree",
            )
            if hasattr(config.parallelism, k)
        },
        "num_tokens_per_microbatch_per_dp_rank": config.training.num_tokens_per_microbatch_per_dp_rank,
        "num_tokens_per_train_step": config.training.num_tokens_per_train_step,
        "steps": config.training.steps,
        "profiled_step": profile_step,
        "model_config": type(config.model).__qualname__,
        # Read by make_problems.py to generate realistic problem inputs.
        "model": _model_constants(config.model),
    }
    with open(os.path.join(out_dir, "setup.json"), "w") as f:
        json.dump(setup, f, indent=1, default=str)
    return config


def _find_config(node: Any, cls: type) -> Any:
    """First instance of ``cls`` in a config tree (dataclasses, lists, dicts), or None."""
    if isinstance(node, cls):
        return node
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        children = [getattr(node, f.name) for f in dataclasses.fields(node)]
    elif isinstance(node, (list, tuple)):
        children = list(node)
    elif isinstance(node, dict):
        children = list(node.values())
    else:
        return None
    for child in children:
        found = _find_config(child, cls)
        if found is not None:
            return found
    return None


def _model_constants(model_config: Any) -> dict[str, Any]:
    router = _find_config(model_config, TokenChoiceTopKRouter.Config)
    return {
        "vocab_size": getattr(model_config, "vocab_size", None),
        "max_context_length": getattr(model_config, "max_context_length", None),
        "router": None
        if router is None
        else {
            k: getattr(router, k, None)
            for k in ("num_experts", "top_k", "num_expert_groups", "num_limited_groups")
        },
    }


def _record_world_size() -> None:
    """Add the world size, known once the process group exists, to setup.json."""
    path = os.path.join(_out_dir(), "setup.json")
    if not torch.distributed.is_initialized() or not os.path.exists(path):
        return
    with open(path) as f:
        setup = json.load(f)
    setup["world_size"] = torch.distributed.get_world_size()
    with open(path, "w") as f:
        json.dump(setup, f, indent=1, default=str)


def _out_dir() -> str:
    out = os.environ.get(_OUT_ENV)
    if not out:
        raise RuntimeError(
            f"{_OUT_ENV} is unset; call enable_fusion_finder() in the recipe"
        )
    return out


def _short(x: Any, limit: int = 200) -> str:
    s = repr(x)
    return s if len(s) <= limit else s[:limit] + "...<trunc>"


def _encode_arg(a: Any) -> Any:
    if isinstance(a, fx.Node):
        return {"node": a.name}
    if isinstance(a, (list, tuple)):
        return [_encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {str(k): _encode_arg(v) for k, v in a.items()}
    if isinstance(a, (int, float, bool, str)) or a is None:
        return a
    if isinstance(
        a,
        (
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
        ),
    ):
        return str(a)
    return {"repr": _short(a)}


def _sizes(xs: Any) -> list[Any]:
    return [int(s) if not isinstance(s, torch.SymInt) else str(s) for s in xs]


def _encode_val(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        try:
            stride = _sizes(v.stride())
        except RuntimeError:  # sparse / nested tensors have no strides
            stride = None
        return {
            "t": "tensor",
            "shape": _sizes(v.shape),
            "stride": stride,
            "dtype": str(v.dtype),
            "device": str(v.device),
        }
    if isinstance(v, (list, tuple)):
        return [_encode_val(x) for x in v]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return {"repr": _short(v)}


def _target_str(node: fx.Node) -> str:
    t = node.target
    if node.op == "call_function":
        if isinstance(t, torch._ops.OpOverload):
            return str(t)
        if isinstance(t, torch._ops.HigherOrderOperator):
            return f"hop.{t.name()}"
        qual = getattr(t, "__qualname__", getattr(t, "__name__", str(t)))
        return f"{getattr(t, '__module__', '')}.{qual}"
    return str(t)


def _node_record(idx: int, node: fx.Node) -> dict[str, Any]:
    meta = node.meta
    # nn_module_stack is usually a dict but some passes leave a string.
    nn_stack = meta.get("nn_module_stack")
    if isinstance(nn_stack, dict):
        nn_stack = [[str(k), str(v[0]), str(v[1])] for k, v in nn_stack.items()]
    elif nn_stack is not None:
        nn_stack = _short(nn_stack, 1000)
    custom = meta.get("custom")
    if isinstance(custom, dict):
        custom = {str(k): _short(v, 400) for k, v in custom.items()}
    elif custom is not None:
        custom = {"_repr": _short(custom, 400)}
    stack_trace = meta.get("stack_trace")
    return {
        "idx": idx,
        "name": node.name,
        "op": node.op,
        "target": _target_str(node),
        # Lets lift_hazard read in-place, out= and view aliasing offline.
        "schema": str(node.target._schema)
        if isinstance(node.target, torch._ops.OpOverload)
        else None,
        "args": _encode_arg(node.args),
        "kwargs": _encode_arg(node.kwargs),
        "users": [u.name for u in node.users],
        "val": _encode_val(meta.get("val")) if "val" in meta else None,
        "stack_trace": stack_trace if isinstance(stack_trace, str) else None,
        "nn_module_stack": nn_stack,
        "custom": custom,
        "recompute": str(meta["recompute"]) if "recompute" in meta else None,
        "autograd_backward": bool(meta.get("autograd_backward", False)),
        "seq_nr": _encode_arg(meta.get("seq_nr")),
    }


def dump_graph_pass(
    gm: fx.GraphModule, example_inputs: Any = None, *, tag: str
) -> fx.GraphModule:
    """Write ``graph_dump/<tag>_nodes.jsonl.gz`` (one JSON record per node) and a readable graph."""
    _record_world_size()
    out = os.path.join(_out_dir(), "graph_dump")
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, f"{tag}_nodes.jsonl.gz"), "wt") as f:
        for idx, node in enumerate(gm.graph.nodes):
            f.write(json.dumps(_node_record(idx, node), default=_short) + "\n")
    text = gm.print_readable(
        print_output=False, include_stride=True, include_device=True, expanded_def=True
    )
    with gzip.open(os.path.join(out, f"{tag}_graph.txt.gz"), "wt") as f:
        f.write(text)
    logger.info(
        "[fusion_finder] dumped %s graph (%d nodes) to %s",
        tag,
        len(gm.graph.nodes),
        out,
    )
    return gm


_FRESH_BUFFER_TARGETS = (
    torch.ops.aten.new_empty.default,
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros.default,
)


def recompute_fresh_buffer_index_put_pass(
    gm: fx.GraphModule, example_inputs: Any = None
) -> fx.GraphModule:
    """Retag ``fresh buffer -> index_put_ -> consumers`` from MUST_SAVE to MUST_RECOMPUTE.

    ``tag_sac_policy`` forces every in-place mutation and its direct forward
    consumers to MUST_SAVE. In the MoE combine path that saves the combine
    all_to_all output ([routed rows, dim] per MoE layer), which runs DeepSeek-V3
    671B out of memory under ``memory_policy="full"``. The buffer is private and
    written once, so recomputing the chain is safe and matches eager full AC.
    Runs after ``tag_with_memory_policy_pass``. Retagged chains are written to
    ``retagged_index_put_nodes.json``.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    fixed = []
    for node in gm.graph.nodes:
        if (
            node.meta.get("autograd_backward")
            or node.target is not torch.ops.aten.index_put_.default
        ):
            continue
        buf = node.args[0]
        if not isinstance(buf, fx.Node) or buf.target not in _FRESH_BUFFER_TARGETS:
            continue
        if (
            any(u is not node for u in buf.users)
            or node.meta.get("recompute") != CheckpointPolicy.MUST_SAVE
        ):
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        buf.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        retagged = [buf.name, node.name]
        stack = list(node.users)
        while stack:
            user = stack.pop()
            if user.meta.get("autograd_backward"):
                continue
            if user.meta.get("recompute") == CheckpointPolicy.MUST_SAVE:
                user.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
                retagged.append(user.name)
            # getitem / wait_tensor inherit their parent's decision.
            for u2 in user.users:
                if u2.target is torch.ops._c10d_functional.wait_tensor.default or (
                    getattr(u2.target, "__name__", "") == "getitem"
                ):
                    stack.append(u2)
        fixed.append(retagged)
    with open(os.path.join(_out_dir(), "retagged_index_put_nodes.json"), "w") as f:
        json.dump(fixed, f, indent=1)
    logger.info(
        "[fusion_finder] retagged %d fresh-buffer index_put_ chains to MUST_RECOMPUTE",
        len(fixed),
    )
    return gm


def _pipeline(
    traced, trainer_config, *, parallelism_context, recompute_index_put: bool
):
    passes = construct_default_graph_passes(
        traced, trainer_config, parallelism_context=parallelism_context
    )
    names = [_get_pass_name(p) for p in passes]
    if recompute_index_put:
        passes.insert(
            names.index("tag_with_memory_policy_pass") + 1,
            recompute_fresh_buffer_index_put_pass,
        )
        names = [_get_pass_name(p) for p in passes]
    # Dump before the final Inductor passes (regional or full), i.e. the aten graph after
    # memory policy, remat, bucketing and reordering.
    inductor_starts = [
        i
        for i, n in enumerate(names)
        if n
        in (
            "annotate_flex_attention_for_regional_inductor_pass",
            "full_inductor_compilation_pass",
        )
    ]
    passes.insert(
        inductor_starts[0] if inductor_starts else len(passes),
        functools.partial(dump_graph_pass, tag="pre_inductor"),
    )
    passes.append(functools.partial(dump_graph_pass, tag="final"))
    passes.append(
        functools.partial(
            custom_codegen_pass, codegen_dir=os.path.join(_out_dir(), "codegen")
        )
    )
    with open(os.path.join(_out_dir(), "pass_list.txt"), "w") as f:
        f.write("\n".join(_get_pass_name(p) for p in passes) + "\n")
    return passes


@register_pass_pipeline("fusion_finder")
def fusion_finder_pipeline(traced, trainer_config, *, parallelism_context):
    return _pipeline(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
        recompute_index_put=False,
    )


@register_pass_pipeline("fusion_finder_recompute_index_put")
def fusion_finder_recompute_index_put_pipeline(
    traced, trainer_config, *, parallelism_context
):
    return _pipeline(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
        recompute_index_put=True,
    )
