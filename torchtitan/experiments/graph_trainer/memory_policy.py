# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory policy passes for graph_trainer.

Selective activation checkpointing (SAC) tagging and memory policy dispatch.
Each saved forward activation can independently be tagged as MUST_SAVE,
MUST_RECOMPUTE, or MUST_CPU_OFFLOAD.  The ``tag_with_memory_policy_pass``
entry point selects a tagging strategy via ``--compile.memory_policy``.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_checkpoint import _get_default_save_ops
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _MODULE_FQN,
    _NOT_IN_LAYERS,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    tag_all_offloadable_activations,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.log_activation_memory_policy import (
    log_activation_memory_policy,
)
from torchtitan.experiments.graph_trainer.registry import (
    MEMORY_POLICY_REGISTRY,
    register_memory_policy,
)
from torchtitan.tools.logging import logger


def _make_default_memory_policy(save_ops: set | None = None) -> Callable:
    """Create a SAC policy function from a set of op targets to save."""
    if save_ops is None:
        save_ops = _get_default_save_ops()

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def _find_fsdp_unshard_save_nodes(gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
    save_nodes: set[torch.fx.Node] = set()
    for node in gm.graph.find_nodes(op="placeholder"):
        save_nodes.update(find_fsdp_unshard_save_nodes(node))
    return save_nodes


def _make_full_memory_policy() -> Callable:
    """Full recompute policy: mark everything as MUST_RECOMPUTE.

    The layer boundary pass in tag_sac_policy will force MUST_SAVE on nodes
    whose output crosses a layer boundary, so only layer outputs are saved.
    This mirrors eager's full AC (checkpoint_wrapper with no context_fn),
    which recomputes the entire block — including attention — in backward.

    RNG ops (dropout etc.) are the one class that is always saved: the remat
    pass cannot replay their random state, and ``has_recomputable_rng_ops``
    would otherwise raise. Eager full AC recomputes them by forking the RNG
    state (``preserve_rng_state=True``); the graph path lacks that, so it
    saves them instead.

    Higher-order ops (e.g. flex_attention) ARE recomputed: ``node_copy``
    duplicates them together with their ``get_attr`` subgraph references, and
    the subsequent regional_inductor pass compiles the duplicate as well.
    """

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", set()):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return policy_fn


def _make_none_memory_policy() -> Callable:
    """No-recompute policy: mark every tagged forward op as MUST_SAVE."""

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        return CheckpointPolicy.MUST_SAVE

    return policy_fn


def _make_eager_memory_policy(save_ops: set | None = None) -> Callable:
    """Eager-compatible SAC policy that alternates mm ops between save/recompute.

    Matches the behavior of torchtitan.distributed.activation_checkpoint:
    every second mm/linear op is marked PREFER_RECOMPUTE instead of MUST_SAVE.
    The mm counter resets at each layer boundary so every layer sees the same
    alternation pattern, just like eager AC's per-layer checkpoint_wrapper.
    """
    if save_ops is None:
        save_ops = _get_default_save_ops()
    mm_ops = {torch.ops.aten.mm.default, torch.ops.aten.linear.default}
    mm_count = 0
    current_layer = None

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        nonlocal mm_count, current_layer
        layer_id = _get_layer_id(node)
        if layer_id != _NOT_IN_LAYERS and layer_id != current_layer:
            mm_count = 0
            current_layer = layer_id

        if node.target in mm_ops:
            mm_count += 1
            if node.target in save_ops and mm_count % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def tag_sac_policy(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    policy_fn: Callable[[torch.fx.Node], CheckpointPolicy] | None = None,
    force_save_nodes: set[torch.fx.Node] | None = None,
) -> torch.fx.GraphModule:
    """Apply selective activation checkpointing on the joint graph.

    Annotates forward ``call_function`` nodes with a ``CheckpointPolicy``
    determined by ``policy_fn``. After tagging, a boundary pass forces
    ``MUST_SAVE`` on recomputable nodes whose output crosses a layer
    boundary (layer N → layer N+1), since recomputing them would require
    rerunning the entire preceding layer.

    ``getitem`` / ``wait_tensor`` nodes inherit the parent's tag.

    The model must have been annotated with ``annotate_module_fqns`` before
    tracing so that nodes carry ``module_fqn`` metadata.

    Args:
        gm: The joint forward-backward graph module.
        policy_fn: Callable that takes a node and returns a CheckpointPolicy.
            Defaults to ``_make_default_memory_policy()`` if None.
        force_save_nodes: Nodes that must be saved independent of ``policy_fn``.
            Used for graph-structure constraints such as FSDP unshards with
            ``reshard_after_forward=False``.

    Returns:
        The annotated graph module
    """
    if policy_fn is None:
        policy_fn = _make_default_memory_policy()
    if force_save_nodes is None:
        force_save_nodes = set()

    # Pass 1: Tag each forward node with a recompute policy.
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        # Skip backward nodes — they must not carry recompute tags,
        # otherwise the remat pass would try to duplicate backward ops.
        if _is_backward_node(node):
            continue

        # Skip the post-layer epilogue (lm_head + loss). Chunked-loss
        # regions split backward into multiple disjoint regions, and the
        # remat pass only supports one region with must_recompute deps.
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
        if fqn.startswith(("lm_head", "loss")):
            continue

        if node in force_save_nodes:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue

        if node.target in (
            operator.getitem,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            # Propagate from parent: getitem extracts tuple elements,
            # wait_tensor is tied to its async collective — both must
            # share the parent's save/recompute decision.
            parent = node.args[0]
            if isinstance(parent, torch.fx.Node) and "recompute" in parent.meta:
                node.meta["recompute"] = parent.meta["recompute"]
            continue

        # Always save sym-int nodes (shape reads like sym_size/sym_stride, and
        # tensor->int scalar conversions) rather than recompute them: recomputing
        # a shape read pins the parent tensor alive in backward just to reread its
        # size. We key off meta["val"] being a SymInt -- mirroring AOT Autograd's
        # partitioner, which saves SymInts (cheap scalars) but never SymFloats.
        if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue

        # NOTE: The eager SAC policy (activation_checkpoint.py) alternates
        # mm ops between MUST_SAVE and PREFER_RECOMPUTE. We omit that here
        # because the alternating heuristic is arbitrary.
        node.meta["recompute"] = policy_fn(node)

    # Pass 2: Force MUST_SAVE at layer boundaries. If a recomputable node
    # feeds into a node in a higher layer, saving it is cheaper than
    # recomputing the entire preceding layer.
    def _is_recomputable(n: torch.fx.Node) -> bool:
        return n.meta.get("recompute") in (
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        )

    boundary_saves = 0
    for node in gm.graph.nodes:
        if _is_backward_node(node) or not _is_recomputable(node):
            continue
        node_layer_id = _get_layer_id(node)
        for user in node.users:
            if (
                not _is_backward_node(user)
                and _is_recomputable(user)
                and _get_layer_id(user) > node_layer_id
            ):
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                boundary_saves += 1
                break

    gm.recompile()

    # Per-layer summary from the FINAL policy (after boundary forcing). Counts
    # every forward node carrying a recompute decision — primary policy-tagged
    # ops plus getitem/wait_tensor (which inherit a parent's tag) plus any node
    # the boundary pass forced to MUST_SAVE — so the per-layer MUST_SAVE counts
    # account for boundary saves wherever they land (including on getitem /
    # wait_tensor). The recompute count covers both PREFER_RECOMPUTE and
    # MUST_RECOMPUTE, so the column is labelled generically as RECOMPUTE.
    layer_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"save": 0, "recompute": 0}
    )
    for node in gm.graph.nodes:
        if "recompute" not in node.meta:
            continue
        key = (
            "save"
            if node.meta["recompute"] == CheckpointPolicy.MUST_SAVE
            else "recompute"
        )
        layer_stats[_get_layer_id(node)][key] += 1

    logger.info("Applied selective activation checkpointing (SAC) graph pass.")
    if boundary_saves:
        logger.info(f"  Forced {boundary_saves} nodes to MUST_SAVE at layer boundaries")
    for layer_id in sorted(layer_stats):
        stats = layer_stats[layer_id]
        label = "non-layer" if layer_id == _NOT_IN_LAYERS else str(layer_id)
        logger.info(
            f"  Layer {label}: "
            f"{stats['save']} MUST_SAVE, "
            f"{stats['recompute']} RECOMPUTE"
        )
    return gm


@register_memory_policy("default")
def _default_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """SAC policy that saves compute-intensive ops and required FSDP unshards."""
    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=config.parallelism.pipeline_parallel_degree > 1,
    )
    force_save_nodes = (
        _find_fsdp_unshard_save_nodes(gm) if not fsdp_reshard_after_forward else None
    )
    tag_sac_policy(
        gm,
        policy_fn=_make_default_memory_policy(),
        force_save_nodes=force_save_nodes,
    )
    return gm


@register_memory_policy("full")
def _full_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute: only layer outputs are saved."""
    tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
    return gm


@register_memory_policy("none")
def _none_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """No recompute: save all tagged forward activations."""
    tag_sac_policy(gm, policy_fn=_make_none_memory_policy())
    return gm


@register_memory_policy("eager")
def _eager_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """SAC policy that alternates mm ops between save/recompute."""
    tag_sac_policy(gm, policy_fn=_make_eager_memory_policy())
    return gm


@register_memory_policy("sac_and_offload")
def _sac_and_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """SAC + CPU offload: apply default SAC, then offload within budget."""
    _default_memory_policy_pass(gm, config=config)
    tag_all_offloadable_activations(
        gm,
        cpu_budget_gb=config.compile.cpu_offload_budget_gb,
    )
    return gm


def tag_with_memory_policy_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Tag forward nodes with MUST_SAVE, PREFER_RECOMPUTE, or MUST_CPU_OFFLOAD.

    The ``config.compile.memory_policy`` selects the tagging strategy:
        default: SAC with all compute-intensive ops saved.
        full: full recompute — only layer outputs are saved.
        none: no recompute — save all tagged forward activations.
        eager: SAC alternating mm ops between save/recompute.
        sac_and_offload: SAC + CPU offload within budget.

    Other memory policies combining SAC and CPU offload can be added
    via ``register_memory_policy`` without modifying this function.
    """
    memory_policy = config.compile.memory_policy
    if memory_policy not in MEMORY_POLICY_REGISTRY:
        raise ValueError(
            f"Unknown memory_policy: {memory_policy!r}. "
            f"Available: {list(MEMORY_POLICY_REGISTRY.keys())}"
        )
    gm = MEMORY_POLICY_REGISTRY[memory_policy](gm, config=config)
    log_activation_memory_policy(gm)
    return gm
