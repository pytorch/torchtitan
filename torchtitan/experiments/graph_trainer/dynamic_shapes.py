# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic-shape helpers used by minimal_fx_tracer.

Covers:
- detection of mark_unbacked() and mark_dynamic() metadata on plain and
  wrapper-subclass tensors,
- input fakeification with a StatelessSymbolicContext built from that metadata,
- post-trace materialization of deferred ShapeEnv runtime asserts and
  symbolic-input guards.

Contract:
- Users express input dynamic shapes with PyTorch's Dynamo tensor-marking APIs:
  ``torch._dynamo.mark_dynamic`` and
  ``torch._dynamo.decorators.mark_unbacked``.
- Marked dims are supported only on plain tensor inputs. Wrapper subclasses are
  rejected before unwrapping so markings cannot be silently dropped or copied to
  the wrong inner tensor.
- This module is the only place where minimal_fx_tracer interprets or copies
  raw ``_dynamo_*`` tensor annotations.
"""

from typing import Any

import torch
from torch._subclasses import FakeTensorMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


_DYNAMO_SHAPE_ANNOTATION_NAMES = (
    "_dynamo_unbacked_indices",
    "_dynamo_strict_unbacked_indices",
    "_dynamo_dynamic_indices",
    "_dynamo_dynamic_range",
    "_dynamo_shape_ids",
    "_dynamo_unbacked_bounds",
    "_dynamo_hint_overrides",
    "_specialize_on",
    "_has_dynamo_dim_marking",
)
"""Dynamo tensor attributes that describe user-requested symbolic shapes.

This is an explicit allowlist, not a general tensor-attribute copy. The tracer
reads a subset of these attrs to build a SymbolicContext, then copies the full
allowlist onto the fake tensor so later Dynamo/FakeTensor consumers see the
same shape annotations they would have seen on the original input.
"""


def _tensor_has_mark_unbacked(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` itself carries mark_unbacked() metadata."""
    return bool(
        getattr(tensor, "_dynamo_unbacked_indices", None)
        or getattr(tensor, "_dynamo_strict_unbacked_indices", None)
    )


def _tensor_has_mark_dynamic(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` itself carries mark_dynamic() metadata."""
    return bool(getattr(tensor, "_dynamo_dynamic_indices", None))


def _wrapper_subclass_has_marked_dynamic_dims(tensor: torch.Tensor) -> bool:
    """Detect marked dims anywhere inside a traceable wrapper subclass.

    minimal_fx_tracer unwraps subclasses such as DTensor before fakeification
    and rewraps them after execution. Marked dynamic dims on wrapper inputs are
    therefore rejected rather than silently dropped or copied onto the wrong
    plain inner tensor.
    """
    if not is_traceable_wrapper_subclass(tensor):
        return False
    if _tensor_has_mark_unbacked(tensor) or _tensor_has_mark_dynamic(tensor):
        return True

    attrs, _ = tensor.__tensor_flatten__()
    for attr in attrs:
        inner_value = getattr(tensor, attr)
        if not isinstance(inner_value, torch.Tensor):
            continue
        if (
            _tensor_has_mark_unbacked(inner_value)
            or _tensor_has_mark_dynamic(inner_value)
            or _wrapper_subclass_has_marked_dynamic_dims(inner_value)
        ):
            return True
    return False


def _symbolic_context_for_marked_dims(tensor: torch.Tensor) -> Any | None:
    """Convert Dynamo shape annotations into FakeTensor symbolic context.

    This function semantically interprets the original tensor's annotations:
    which dims are unbacked, dynamic, range-constrained, or specialization
    candidates. It deliberately uses direct ``getattr`` reads instead of the
    copy allowlist because each attribute affects the SymbolicContext in a
    different way.
    """
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        RelaxedUnspecConstraint,
        StatelessSymbolicContext,
        StrictMinMaxConstraint,
    )
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.value_ranges import ValueRanges

    marked_unbacked_indices = getattr(tensor, "_dynamo_unbacked_indices", set())
    marked_strict_unbacked_indices = getattr(
        tensor, "_dynamo_strict_unbacked_indices", set()
    )
    marked_dynamic_indices = getattr(tensor, "_dynamo_dynamic_indices", set())
    # mark_dynamic() records ranges separately from the marked dim set. The
    # dim set says a dimension is dynamic; the range refines its constraints.
    dynamic_ranges = {
        dim_range.dim: dim_range
        for dim_range in getattr(tensor, "_dynamo_dynamic_range", set())
    }
    has_outer_marked_dims = bool(
        marked_unbacked_indices
        or marked_strict_unbacked_indices
        or marked_dynamic_indices
    )
    if not has_outer_marked_dims:
        return None

    dynamic_sizes = [DimDynamic.STATIC] * tensor.dim()
    constraint_sizes: list[Any] = [None] * tensor.dim()
    specialize_on = [
        list(getattr(tensor, "_specialize_on", {}).get(i, []))
        for i in range(tensor.dim())
    ]

    for dim in range(tensor.dim()):
        if dim in marked_unbacked_indices:
            dynamic_sizes[dim] = DimDynamic.UNBACKED
        elif dim in marked_strict_unbacked_indices:
            dynamic_sizes[dim] = DimDynamic.UNBACKED
            constraint_sizes[dim] = RelaxedUnspecConstraint(warn_only=False)
        elif dim in marked_dynamic_indices:
            dynamic_sizes[dim] = DimDynamic.DYNAMIC
            dim_range = dynamic_ranges.get(dim)
            if dim_range is None or (dim_range.min is None and dim_range.max is None):
                constraint_sizes[dim] = RelaxedUnspecConstraint(warn_only=False)
            else:
                # mark_dynamic() stores one-sided ranges with None, while
                # ValueRanges requires concrete sentinels. ShapeEnv still
                # intersects this with its normal dynamic-size invariant later.
                constraint_sizes[dim] = StrictMinMaxConstraint(
                    vr=ValueRanges(
                        lower=0 if dim_range.min is None else dim_range.min,
                        upper=int_oo if dim_range.max is None else dim_range.max,
                    ),
                    warn_only=False,
                )

    return StatelessSymbolicContext(
        dynamic_sizes=dynamic_sizes,
        constraint_sizes=constraint_sizes,
        specialize_on=specialize_on,
        shape_ids=getattr(tensor, "_dynamo_shape_ids", None),
        unbacked_bounds=getattr(tensor, "_dynamo_unbacked_bounds", None),
    )


def _fakeify_input(
    fake_mode: FakeTensorMode, arg: torch.Tensor, *, input_name: str
) -> torch.Tensor:
    """Fakeify one tracer input while preserving user shape annotations.

    ``FakeTensorMode.from_tensor`` consumes the SymbolicContext but does not
    generally preserve Python attributes from ``arg``. After fakeification, copy
    only Dynamo's shape-annotation allowlist so later graph passes and Dynamo
    helpers can still inspect the fake tensor consistently.
    """
    from torch._dynamo.source import LocalSource

    def copy_tensor_annotations(fake_arg: torch.Tensor) -> torch.Tensor:
        for name in _DYNAMO_SHAPE_ANNOTATION_NAMES:
            if hasattr(arg, name):
                setattr(fake_arg, name, getattr(arg, name))
        return fake_arg

    symbolic_context = _symbolic_context_for_marked_dims(arg)
    if symbolic_context is None:
        return copy_tensor_annotations(fake_mode.from_tensor(arg, static_shapes=True))

    source = LocalSource(input_name, is_input=True)
    # Do not wrap this in ShapeEnv.ignore_fresh_unbacked_symbols(): the
    # StatelessSymbolicContext binds marked input dims to placeholders, so they
    # are not unowned fresh symbols produced by tracing graph operations.
    return copy_tensor_annotations(
        fake_mode.from_tensor(
            arg,
            source=source,
            symbolic_context=symbolic_context,
        )
    )


def _insert_runtime_asserts(
    gm: torch.fx.GraphModule, fake_mode: FakeTensorMode
) -> None:
    """Materialize the ShapeEnv's deferred runtime asserts into the graph.

    Constraints accumulated during fake-tensor propagation (e.g. ``u0 >= 0``,
    ``torch._check(...)``) only live in the ShapeEnv. Without this pass they
    are silently dropped at runtime, so unbacked-symbol assumptions go
    unchecked. Mirrors the call Dynamo and HOP-subgraph proxy tracing make
    after capture.
    """
    if fake_mode.shape_env is None:
        return
    from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

    insert_deferred_runtime_asserts(gm, fake_mode.shape_env, "minimal_fx_tracer")
    gm.recompile()
