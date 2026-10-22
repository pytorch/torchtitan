# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local backports for the PyTorch 2.15 release candidate."""

import hashlib
import importlib
import inspect
import linecache
import logging
import os
import textwrap
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_applied_patches: tuple[int, ...] | None = None


def apply_patches() -> tuple[int, ...]:
    """Apply verified Python backports without changing installed torch files.

    Import TorchTitan before constructing pipeline schedules. Set
    TORCHTITAN_DISABLE_TORCH_COMPAT=1 before import to run the unpatched RC.
    Use a fresh process when changing this setting.
    """
    global _applied_patches
    if _applied_patches is not None:
        return _applied_patches
    if os.environ.get("TORCHTITAN_DISABLE_TORCH_COMPAT") == "1":
        return ()
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        return ()
    if torch_version.split("+")[0] != "2.15.0":
        return ()

    from torch.distributed.pipelining import _backward, schedules, stage
    from torch.utils.hooks import RemovableHandle

    from ._pipelining_edits import PATCHES

    needed = {
        199884: not hasattr(
            schedules._PipelineSchedule, "register_post_metadata_inference_cleanup"
        ),
        200238: any(
            "loss_kwarg_mbs" not in inspect.signature(cls.step).parameters
            for cls in (
                schedules.PipelineScheduleSingle,
                schedules.PipelineScheduleMulti,
            )
        ),
        200199: "not stage_output._is_view()"
        not in inspect.getsource(_backward.stage_backward_input),
    }
    # Prepare every change before publishing any of them. An unrecognized RC
    # must fail explicitly rather than leave a partially patched torch module.
    pending: dict[tuple[Any, str], Any] = {}
    sources: dict[tuple[Any, str], str] = {}
    applied: list[int] = []
    for patch in PATCHES:
        pr = patch["pr"]
        if not needed[pr]:
            continue
        module = importlib.import_module(patch["module"])
        for edit in patch["functions"]:
            target = edit["target"]
            parent, _, name = target.rpartition(".")
            owner = getattr(module, parent) if parent else module
            key = (owner, name)
            if "source" in edit:
                source = edit["source"]
            else:
                original = pending[key] if key in pending else getattr(owner, name)
                if original.__closure__:
                    raise RuntimeError(f"Cannot backport a closure: {target}")
                source = sources.get(key)
                if source is None:
                    source = textwrap.dedent(inspect.getsource(original))
                if hashlib.sha256(source.encode()).hexdigest() != edit["before_sha256"]:
                    raise RuntimeError(
                        f"Unsupported torch {torch_version} source for {target}; "
                        f"revalidate compatibility backport #{pr} for this RC."
                    )
                for before, after in edit["edits"]:
                    if source.count(before) != 1:
                        raise RuntimeError(f"Backport #{pr} context mismatch: {target}")
                    source = source.replace(before, after, 1)
                if hashlib.sha256(source.encode()).hexdigest() != edit["after_sha256"]:
                    raise RuntimeError(f"Backport #{pr} result mismatch: {target}")

            filename = f"{__file__}:pytorch-{pr}:{target}"
            namespace = dict(vars(module))
            namespace.update(OrderedDict=OrderedDict, RemovableHandle=RemovableHandle)
            exec(compile(source, filename, "exec"), namespace)
            replacement = namespace[name]
            replacement.__qualname__ = target
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(keepends=True),
                filename,
            )
            pending[key] = replacement
            sources[key] = source
        applied.append(pr)

    for (owner, name), replacement in pending.items():
        setattr(owner, name, replacement)
    if 200199 in applied:
        stage.stage_backward_input = _backward.stage_backward_input
    _applied_patches = tuple(applied)
    if applied:
        logging.getLogger(__name__).warning(
            "Applied TorchTitan process-local compatibility backports %s to torch %s",
            applied,
            torch_version,
        )
    return _applied_patches
