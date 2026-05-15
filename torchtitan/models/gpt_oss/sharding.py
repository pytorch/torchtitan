# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPT-OSS sharding scaffold.

The previous implementation was intentionally removed so autoresearch cannot
copy an existing TorchTitan model sharding strategy.
"""


def set_gpt_oss_sharding_config(
    config,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    raise NotImplementedError("NotImplemented")
