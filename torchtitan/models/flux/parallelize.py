# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux parallelization scaffold.

The previous implementation was intentionally removed so autoresearch cannot
copy an existing TorchTitan model parallelization strategy.
"""

__all__ = [
    "apply_ac",
    "apply_compile",
    "apply_cp",
    "apply_fsdp",
    "parallelize_encoders",
    "parallelize_flux",
]


def parallelize_flux(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    raise NotImplementedError("NotImplemented")


def apply_fsdp(*args, **kwargs):
    raise NotImplementedError("NotImplemented")


def apply_compile(*args, **kwargs):
    raise NotImplementedError("NotImplemented")


def apply_ac(*args, **kwargs):
    raise NotImplementedError("NotImplemented")


def apply_cp(*args, **kwargs):
    raise NotImplementedError("NotImplemented")


def parallelize_encoders(*args, **kwargs):
    raise NotImplementedError("NotImplemented")
