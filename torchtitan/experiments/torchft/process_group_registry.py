# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

ProcessGroupFactory = Callable[[timedelta], Any]

_PROCESS_GROUP_FACTORIES: dict[str, ProcessGroupFactory] = {}


def register_process_group_factory(
    name: str,
    factory: ProcessGroupFactory,
) -> None:
    """Register a TorchFT process-group factory.

    Re-registering the same factory is a no-op. Registering a different
    factory under an existing name is rejected to avoid silently replacing a
    backend selected by the training configuration.
    """
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Process group name must not be empty")
    if not callable(factory):
        raise TypeError("Process group factory must be callable")

    existing = _PROCESS_GROUP_FACTORIES.get(normalized_name)
    if existing is factory:
        return
    if existing is not None:
        raise ValueError(
            f"Process group {normalized_name!r} is already registered"
        )

    _PROCESS_GROUP_FACTORIES[normalized_name] = factory


def create_process_group(name: str, timeout: timedelta) -> Any:
    """Create a registered TorchFT process group."""
    normalized_name = name.strip().lower()
    factory = _PROCESS_GROUP_FACTORIES.get(normalized_name)
    if factory is None:
        registered = ", ".join(sorted(_PROCESS_GROUP_FACTORIES))
        raise ValueError(
            f"Unsupported process group: {name}. "
            f"Registered process groups: {registered}"
        )
    return factory(timeout)


def registered_process_group_names() -> tuple[str, ...]:
    """Return registered process-group names in deterministic order."""
    return tuple(sorted(_PROCESS_GROUP_FACTORIES))


def _create_gloo(timeout: timedelta) -> Any:
    import torchft

    return torchft.ProcessGroupGloo(timeout=timeout)


def _create_nccl(timeout: timedelta) -> Any:
    import torchft

    return torchft.ProcessGroupNCCL(timeout=timeout)


def _create_mccl(timeout: timedelta) -> Any:
    import torch
    import torchcomms
    from torchft.torchcomms import ProcessGroupTorchComms

    comm = torchcomms.new_comm(
        "mccl",
        device=torch.device("cuda"),
        name="mccl_ft",
        timeout=timeout,
        enable_reconfigure=True,
    )
    return ProcessGroupTorchComms(comm, timeout=timeout)


register_process_group_factory("gloo", _create_gloo)
register_process_group_factory("nccl", _create_nccl)
register_process_group_factory("mccl", _create_mccl)


__all__ = [
    "ProcessGroupFactory",
    "create_process_group",
    "registered_process_group_names",
    "register_process_group_factory",
]
