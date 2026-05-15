# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPT-OSS parallelization scaffold.

The previous implementation was intentionally removed so autoresearch cannot
copy an existing TorchTitan model parallelization strategy.
"""

__all__ = ["apply_moe_ep_tp", "parallelize_gptoss"]


def parallelize_gptoss(
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


def apply_moe_ep_tp(*args, **kwargs):
    raise NotImplementedError("NotImplemented")
