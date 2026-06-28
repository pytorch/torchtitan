# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph cleanup passes for traced forward-backward graphs.

These passes simplify traced graphs by eliminating nodes that are identity
operations in the context of a fully traced graph (no autograd, no symbolic
shape changes) and by canonicalizing equivalent view ops to a single target.
Removing/normalizing them reduces graph noise and improves downstream pass
effectiveness (bucketing, scheduling, cudagraph compatibility).

``canonicalize_graph_pass`` bundles every individual sub-pass here into a
single pass-list entry; the sub-passes remain public so they can be tested
(and reasoned about) in isolation.
"""

import sys

import torch
from torch.fx.experimental.symbolic_shapes import guard_or_false

from torchtitan.tools.logging import logger

# Op overloads that are registered side-effectful but that we want DCE to treat
# as pure (so unused instances, and their now-orphaned input chains, are dropped).
# Currently just the ``aten._assert_async`` runtime asserts; add other removable
# side-effect ops here as they come up.
_FORCE_PURE_TARGETS = (
    torch.ops.aten._assert_async.msg,
    torch.ops.aten._assert_async.default,
)


def _is_impure_for_dce(node: torch.fx.Node) -> bool:
    """``is_impure_node`` for DCE that forces :data:`_FORCE_PURE_TARGETS` pure.

    Mirrors the default ``node.is_impure()`` for every node except the targets in
    :data:`_FORCE_PURE_TARGETS`, which we report as pure so dead-code elimination
    can drop them (and their now-orphaned input chains). See
    ``eliminate_dead_code_pass``.
    """
    if node.op == "call_function" and node.target in _FORCE_PURE_TARGETS:
        return False
    return node.is_impure()


def eliminate_dead_code_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove dead nodes -- no users and no side effects -- from the graph.

    Delegates to FX's ``Graph.eliminate_dead_code``, which preserves *impure*
    nodes (in-place mutations, ``copy_``, collectives -- anything for which
    ``node.is_impure()`` is True), so only genuinely unused pure computation is
    dropped. Running it first shrinks the graph for every downstream pass (memory
    policy, bucketing, cudagraph partitioning), and removes orphaned subtrees left
    by tracing so they don't get scheduled or counted.

    Dead ``aten._assert_async`` runtime asserts are dropped too, via the custom
    :func:`_is_impure_for_dce` (default DCE keeps them, and their condition chain,
    as side-effectful). They are pure runtime guards with no bearing on numerics.

    Keeping a side-effecting custom op alive: a custom op with a real side effect
    but no users (e.g. a debug/log op, a barrier, an in-place buffer write) is
    *dead* to FX and will be dropped here unless it declares itself impure.
    ``node.is_impure()`` (``torch._library.utils.is_impure``) treats an op's
    ``call_function`` node as impure when any of:

    * Its schema mutates an input -- the natural choice when the op really writes
      a tensor. Declare it via
      ``torch.library.custom_op(..., mutates_args={"buf"})`` (or
      ``mutates_args="unknown"`` as a catch-all), which makes the arg ``Tensor(a!)``
      so ``schema.is_mutable`` is True.
    * It is registered as effectful -- the right choice for a pure side effect with
      no tensor to mutate:
      ``torch.library._register_effectful_op(op, EffectType.ORDERED)`` (from
      ``torch._library.effects``). This also keeps the op *ordered* (token-threaded)
      in functionalized/export graphs, not just un-DCE'd.
    * It is in ``torch.fx.node._side_effectful_functions`` -- the private set used
      by aten asserts / ``record_function``; direct but lowest-level, and gives
      none of the functionalization/ordering guarantees of the above.

    Random ops are also kept (``op._nondeterministic_seeded``). Note that marking
    an op impure also blocks reordering/CSE on it, which is usually the intent for
    a side effect.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with dead code removed.
    """
    if gm.graph.eliminate_dead_code(is_impure_node=_is_impure_for_dce):
        gm.graph.lint()
        gm.recompile()
        logger.info("Eliminated dead code from the graph")
    return gm


def remove_detach_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove ``aten.detach.default`` nodes from the graph.

    In a traced fwd+bwd graph there is no autograd context, so detach is
    semantically a no-op.  Removing these nodes simplifies the graph for
    downstream passes.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with all detach nodes removed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target is torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} aten.detach.default node(s) from the graph")

    return gm


_IDENTITY_VIEW_TARGETS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
}


def _same_shape(shape1: torch.Size, shape2: torch.Size) -> bool:
    return len(shape1) == len(shape2) and all(
        guard_or_false(dim1 == dim2) for dim1, dim2 in zip(shape1, shape2)
    )


def remove_identity_view_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity ``view``, ``reshape``, and ``_unsafe_view`` nodes.

    In a traced graph these ops are no-ops when the output shape equals
    the input shape.  Removing them simplifies the graph for downstream
    passes (bucketing, scheduling, cudagraph).

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with identity view nodes removed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _IDENTITY_VIEW_TARGETS:
            continue

        # Skip nodes without fake tensor metadata (e.g. symbolic or opaque).
        inp = node.args[0]
        inp_val = inp.meta.get("val") if isinstance(inp, torch.fx.Node) else None
        out_val = node.meta.get("val")
        if inp_val is None or out_val is None:
            continue
        if not isinstance(inp_val, torch.Tensor) or not isinstance(
            out_val, torch.Tensor
        ):
            continue

        if _same_shape(inp_val.shape, out_val.shape):
            node.replace_all_uses_with(inp)
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} identity view/reshape node(s) from the graph")

    return gm


def remove_b2b_transpose_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove back-to-back ``t(t(x))`` transposes that cancel out.

    ``aten.t`` on a (<=2-D) tensor swaps the two dimensions.  Applying it
    twice yields the original tensor (identical shape and strides), so the
    back-to-back pair is a no-op regardless of shape.  These pairs appear in
    traced graphs from ``F.linear`` when FSDP redistributes weight tensors.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with back-to-back transpose pairs collapsed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        # A node whose target is the ``aten.t.default`` OpOverload is always a
        # call_function node, so no explicit node.op check is needed.
        if node.target is not torch.ops.aten.t.default:
            continue
        inp = node.args[0]
        if isinstance(inp, torch.fx.Node) and inp.target is torch.ops.aten.t.default:
            original = inp.args[0]
            node.replace_all_uses_with(original)
            gm.graph.erase_node(node)
            count += 1
            # The inner transpose may still feed other consumers; only erase
            # it once it has no remaining users.
            if not inp.users:
                gm.graph.erase_node(inp)
                count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} back-to-back transpose node(s) from the graph")

    return gm


def remove_identity_slice_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity aten.slice.Tensor ops that select the full dimension.

    An ``aten.slice.Tensor(input, dim, start, end, step)`` is a no-op when
    ``start == 0``, ``end >= dim_size``, and ``step == 1``.  This pass
    replaces such nodes with their input tensor, reducing graph noise from
    decompositions and simplifying downstream passes.

    Default args for ``aten.slice.Tensor``: dim=0, start=0, end=sys.maxsize,
    step=1.
    """
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is not torch.ops.aten.slice.Tensor:
            continue

        args = node.args
        input_node = args[0]

        # Parse args with defaults matching aten.slice.Tensor signature
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else sys.maxsize
        step = args[4] if len(args) > 4 else 1

        # Skip if start/end/step are dynamic graph values (FX Nodes) — we
        # can't prove the slice is identity without concrete/symbolic values
        # to compare.
        if (
            isinstance(start, torch.fx.Node)
            or isinstance(end, torch.fx.Node)
            or isinstance(step, torch.fx.Node)
        ):
            continue

        if start != 0 or step != 1:
            continue

        # Use fake tensor metadata to determine the actual dimension size.
        # Skip nodes without metadata (e.g. from hand-built test graphs).
        val = input_node.meta.get("val")
        if val is None:
            continue

        shape = val.shape
        dim_size = shape[dim]

        if guard_or_false(end >= dim_size):
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        logger.info(f"Removed {count} identity slice node(s)")

    gm.graph.lint()
    gm.recompile()
    return gm


def normalize_view_ops_as_reshape(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Retarget ``aten.view`` and ``aten._unsafe_view`` to ``aten.reshape``.

    These three ops are equivalent on contiguous tensors; downstream passes
    pattern-match on ``aten.reshape.default``, so this canonicalizes the
    others to that single target.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with view ops normalized to reshape.
    """
    view_targets = {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_targets:
            node.target = torch.ops.aten.reshape.default
            count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Normalized {count} view op(s) to reshape")

    return gm


def _node_referenced_tensors(node: torch.fx.Node) -> list[torch.Tensor]:
    """FakeTensors referenced by a node's output and its inputs (meta['val'])."""
    # Lazy import avoids a load-time dependency on cudagraph (which itself is
    # imported by passes.py alongside this module).
    from torchtitan.experiments.graph_trainer.cudagraph import _iter_tensors

    tensors = list(_iter_tensors(node.meta.get("val")))
    for inp in node.all_input_nodes:
        tensors += _iter_tensors(inp.meta.get("val"))
    return tensors


def _is_pure_cpu_op(node: torch.fx.Node) -> bool:
    """True if ``node`` is a call_function whose every referenced tensor is CPU.

    Mirrors the pure-CPU rule in ``cudagraph.is_cudagraphable``: a node whose
    output and all inputs are CPU FakeTensors cannot be captured by a CUDA
    graph. Used here to find the dead CPU clusters that gate whole-graph
    cudagraph capture.
    """
    if node.op != "call_function":
        return False
    tensors = _node_referenced_tensors(node)
    return bool(tensors) and all(t.device.type == "cpu" for t in tensors)


def remove_dead_cpu_inplace_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Drop dead pure-CPU in-place ops whose mutation is never observed.

    ``eliminate_dead_code_pass`` preserves in-place (impure) nodes, so a dead
    in-place pure-CPU op -- e.g. the ``aten.add_.Tensor(empty_strided,
    empty_strided_1)`` cluster left by DTensor loss-gradient redistribution
    under make_fx (see ``docs/graph_trainer_design_notes.md`` 11.5.7) --
    survives DCE and gates the all-or-nothing cudagraph capture even though
    nothing reads its result.

    Such a node is safe to erase when ALL hold:

    * It is a pure-CPU ``call_function`` (output + inputs all on CPU).
    * It has no users (its mutated result is never read).
    * Its mutated ``self`` argument (``args[0]``) is a local ``call_function``
      node -- NOT a graph ``placeholder`` -- whose only user is this node.
      This guarantees the in-place mutation is unobserved: no other node reads
      the buffer, and the buffer is not a graph input/accum_grads whose value
      escapes the graph.

    The orphaned allocation inputs (``empty_strided`` etc.) are then dropped by
    a follow-up DCE. The pass is numerics-preserving: it only removes ops whose
    effect is provably unobservable.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with dead pure-CPU in-place clusters removed.
    """
    removed: list[torch.fx.Node] = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or not node.is_impure():
            continue
        if not _is_pure_cpu_op(node):
            continue
        if len(node.users) != 0:
            continue
        # The mutated self must be the first positional arg.
        if not node.args:
            continue
        self_arg = node.args[0]
        if not isinstance(self_arg, torch.fx.Node):
            continue
        # Never touch mutations of graph inputs (placeholders) or of tensors
        # read by any other node: the mutation could be observable.
        if self_arg.op == "placeholder":
            continue
        if len(self_arg.users) != 1:
            continue
        # self_arg's only user is this node; safe to drop both after the
        # in-place op is gone.
        gm.graph.erase_node(node)
        removed.append(node)

    if removed:
        # Drop the now-orphaned allocation chains (empty_strided, etc.).
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        logger.info(
            f"Removed {len(removed)} dead pure-CPU in-place op(s) "
            f"({[n.target.__name__ if hasattr(n.target, '__name__') else str(n.target) for n in removed]})"
        )
    return gm


def canonicalize_graph_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Canonicalize and simplify the graph as a single pass-list entry.

    Bundles the numerics-preserving simplification passes so the pass list
    in ``compile_time_passes`` has one entry instead of several:

    1. ``remove_detach_pass`` — drop autograd-only ``detach`` nodes.
    2. ``remove_identity_view_pass`` — drop ``view``/``reshape``/
       ``_unsafe_view`` nodes whose output shape equals their input.
    3. ``remove_b2b_transpose_pass`` — collapse back-to-back ``t(t(x))`` pairs.
    4. ``remove_identity_slice_pass`` — drop full-dimension slices.
    5. ``normalize_view_ops_as_reshape`` — canonicalize remaining
       ``view``/``_unsafe_view`` nodes to ``reshape``.
    6. ``remove_dead_cpu_inplace_pass`` — drop dead pure-CPU in-place clusters
       that gate whole-graph cudagraph capture (see design notes 11.5.7).

    Each sub-pass lints and recompiles internally, so no extra work is
    needed here. The ordering is irrelevant for correctness (the sub-passes
    act on disjoint op sets, and identity-view removal also handles
    ``reshape``), but is kept stable for readable tlparse diffs.
    """
    gm = remove_detach_pass(gm, example_inputs)
    gm = remove_identity_view_pass(gm, example_inputs)
    gm = remove_b2b_transpose_pass(gm, example_inputs)
    gm = remove_identity_slice_pass(gm, example_inputs)
    gm = normalize_view_ops_as_reshape(gm, example_inputs)
    gm = remove_dead_cpu_inplace_pass(gm, example_inputs)
    return gm
