# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama3 parallelization scaffold.

The previous implementation was intentionally removed so autoresearch cannot
copy an existing TorchTitan model parallelization strategy.
"""

from torchtitan.distributed.fsdp import disable_fsdp_gradient_division

__all__ = ["apply_fsdp", "disable_fsdp_gradient_division", "parallelize_llama"]


def parallelize_llama(
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
