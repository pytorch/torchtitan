# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic-shape helpers used by minimal_fx_tracer.

Covers:
- detection of mark_unbacked() metadata on plain and wrapper-subclass tensors,
- input fakeification with a StatelessSymbolicContext built from that metadata,
- post-trace materialization of deferred ShapeEnv runtime asserts and
  symbolic-input guards.
"""

from typing import Any

import torch
from torch._subclasses import FakeTensorMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


def _tensor_has_mark_unbacked(tensor: torch.Tensor) -> bool:
    return bool(
        getattr(tensor, "_dynamo_unbacked_indices", None)
        or getattr(tensor, "_dynamo_strict_unbacked_indices", None)
    )


def _tensor_has_mark_dynamic(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "_dynamo_dynamic_indices", None))


def _wrapper_subclass_has_marked_dynamic_dims(tensor: torch.Tensor) -> bool:
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
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        RelaxedUnspecConstraint,
        StatelessSymbolicContext,
    )

    marked_unbacked_indices = getattr(tensor, "_dynamo_unbacked_indices", set())
    marked_strict_unbacked_indices = getattr(
        tensor, "_dynamo_strict_unbacked_indices", set()
    )
    marked_dynamic_indices = getattr(tensor, "_dynamo_dynamic_indices", set())
    has_outer_marked_dims = bool(
        marked_unbacked_indices
        or marked_strict_unbacked_indices
        or marked_dynamic_indices
    )
    if not has_outer_marked_dims:
        return None

    dynamic_sizes = [DimDynamic.STATIC] * tensor.dim()
    constraint_sizes = [None] * tensor.dim()
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
    from torch._dynamo.source import LocalSource

    tensor_annotation_names = (
        "_dynamo_unbacked_indices",
        "_dynamo_strict_unbacked_indices",
        "_dynamo_dynamic_indices",
        "_dynamo_shape_ids",
        "_dynamo_unbacked_bounds",
        "_dynamo_hint_overrides",
        "_specialize_on",
    )

    def copy_tensor_annotations(fake_arg: torch.Tensor) -> torch.Tensor:
        for name in tensor_annotation_names:
            if hasattr(arg, name):
                setattr(fake_arg, name, getattr(arg, name))
        return fake_arg

    symbolic_context = _symbolic_context_for_marked_dims(arg)
    if symbolic_context is None:
        return copy_tensor_annotations(fake_mode.from_tensor(arg, static_shapes=True))

    source = LocalSource(input_name, is_input=True)
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
