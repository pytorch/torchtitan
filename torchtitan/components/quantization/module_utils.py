# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Temporary utilities for preserving the Module protocol through quantization conversions.

TODO: These utilities should be removed once torchao natively preserves
the Module protocol (and custom attributes) during quantization conversion.
See https://github.com/pytorch/torchtitan/pull/2527 for context.
"""

from typing import Any

import torch.nn as nn

from torchtitan.protocols.module import Module


def capture_module_attrs(
    model: nn.Module,
    attr_names: list[str],
    nn_module_cls: type[nn.Module] = nn.Linear,
) -> dict[str, dict[str, Any]]:
    """Capture specified attributes from all Module-aware instances.

    Quantization conversion may create new instances or swap classes, losing
    instance attributes.  This captures them by FQN so they can be restored
    afterwards via ``inject_module_protocol(..., saved_attrs=...)``.

    Only modules that are instances of *nn_module_cls* *and* possess **all**
    of the requested attributes are included in the result.
    """
    result: dict[str, dict[str, Any]] = {}
    for fqn, mod in model.named_modules():
        if isinstance(mod, nn_module_cls) and all(
            hasattr(mod, name) for name in attr_names
        ):
            result[fqn] = {name: getattr(mod, name) for name in attr_names}
    return result


def inject_module_protocol(
    model: nn.Module,
    module_cls: type[Module],
    saved_attrs: dict[str, dict[str, Any]] | None = None,
    nn_module_cls: type[nn.Module] = nn.Linear,
) -> None:
    """Inject a ``Module`` subclass into converted modules and restore attrs.

    This API is designed for model converters that may swap the modules (
    mostly nn.Linear) with other classes. For example, after quantization
    conversion, classes like ``Float8Linear`` or ``MXLinear`` inherit from their
    base ``nn.Module`` subclass but not ``torchtitan.models.common.linear.Linear``.
    This API dynamically creates a patched class that inherits from both the converted
    class and *module_cls*, then swaps ``mod.__class__``.

    Args:
        model: The root module to walk.
        module_cls: The ``Module`` subclass to inject (e.g. ``Linear``).
        saved_attrs: Optional ``{fqn: {attr_name: value}}`` dict to restore on
            each matched instance (see ``capture_module_attrs``).
        nn_module_cls: The ``nn.Module`` subclass to match against
            (default ``nn.Linear``).
    """
    patched_classes: dict[type, type] = {}

    def _raise_error(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is a patched class created by "
            f"inject_module_protocol and should not be constructed directly."
        )

    for fqn, mod in model.named_modules():
        if isinstance(mod, nn_module_cls) and not isinstance(mod, Module):
            orig_cls = type(mod)
            if orig_cls not in patched_classes:
                patched_classes[orig_cls] = type(
                    f"{orig_cls.__name__}_With{module_cls.__name__}",
                    (orig_cls, module_cls),
                    {"__init__": _raise_error},
                )
            mod.__class__ = patched_classes[orig_cls]
            if saved_attrs and fqn in saved_attrs:
                for attr_name, value in saved_attrs[fqn].items():
                    setattr(mod, attr_name, value)


def verify_module_protocol(
    model: nn.Module,
    target_cls: type[nn.Module],
    module_cls: type[Module] = Module,
) -> None:
    """Verify that all instances of *target_cls* satisfy the Module protocol.

    Raises ``RuntimeError`` if any module that is an instance of *target_cls*
    is not also an instance of *module_cls*.  This provides a safety net after
    ``inject_module_protocol``: if any target module was missed, the error is
    loud and immediate rather than a silent ``AttributeError`` later.

    Args:
        model: The root module to walk.
        target_cls: The ``nn.Module`` subclass to check (e.g. ``nn.Linear``).
        module_cls: The ``Module`` subclass they must also be (defaults to
            ``Module``; callers can pass ``Linear``).
    """
    failures: list[tuple[str, type]] = []
    for fqn, mod in model.named_modules():
        if isinstance(mod, target_cls) and not isinstance(mod, module_cls):
            failures.append((fqn, type(mod)))

    if failures:
        details = ", ".join(f"'{fqn}' ({cls.__name__})" for fqn, cls in failures)
        raise RuntimeError(
            f"The following {target_cls.__name__} modules do not satisfy "
            f"{module_cls.__name__} protocol: {details}"
        )
