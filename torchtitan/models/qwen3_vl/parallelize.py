# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-VL parallelization scaffold.

The previous implementation was intentionally removed so autoresearch cannot
copy an existing TorchTitan model parallelization strategy.
"""

__all__ = ["parallelize_qwen3_vl"]


def parallelize_qwen3_vl(
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
