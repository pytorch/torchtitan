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


def _is_mutated(node: torch.fx.Node) -> bool:
    """True if any user mutates ``node`` in place (a write-aliased arg position)."""
    for user in node.users:
        schema = getattr(user.target, "_schema", None)
        if schema is None:
            continue
        for i, arg in enumerate(user.args):
            if arg is node and i < len(schema.arguments):
                alias = schema.arguments[i].alias_info
                if alias is not None and alias.is_write:
                    return True
    return False


def remove_lift_fresh_copy_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Drop ``aten.lift_fresh_copy`` clones of a baked tensor constant.

    ``lift_fresh_copy`` clones a freshly-lifted constant so a downstream in-place op
    can mutate the copy without touching the constant. When the input is a baked
    tensor constant (a ``get_attr``) and the copy is never mutated -- e.g. it is only
    read as the value of an ``index_put_`` (a masked ``tensor[mask] = const``) -- the
    clone does nothing, so consumers can read the constant directly. The
    ``_is_mutated`` guard keeps this numerics-preserving (removing a mutated copy
    would clobber the shared constant).

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with redundant lift_fresh_copy nodes removed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.target is not torch.ops.aten.lift_fresh_copy.default:
            continue
        inp = node.args[0]
        if not (isinstance(inp, torch.fx.Node) and inp.op == "get_attr"):
            continue
        if _is_mutated(node):
            continue
        node.replace_all_uses_with(inp)
        gm.graph.erase_node(node)
        count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} lift_fresh_copy node(s) on tensor constants")

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
    5. ``remove_lift_fresh_copy_pass`` — drop ``lift_fresh_copy`` clones of an
       unmutated tensor constant.
    6. ``normalize_view_ops_as_reshape`` — canonicalize remaining
       ``view``/``_unsafe_view`` nodes to ``reshape``.

    Each sub-pass lints and recompiles internally, so no extra work is
    needed here. The ordering is irrelevant for correctness (the sub-passes
    act on disjoint op sets, and identity-view removal also handles
    ``reshape``), but is kept stable for readable tlparse diffs.
    """
    gm = remove_detach_pass(gm, example_inputs)
    gm = remove_identity_view_pass(gm, example_inputs)
    gm = remove_b2b_transpose_pass(gm, example_inputs)
    gm = remove_identity_slice_pass(gm, example_inputs)
    gm = remove_lift_fresh_copy_pass(gm, example_inputs)
    gm = normalize_view_ops_as_reshape(gm, example_inputs)
    return gm
