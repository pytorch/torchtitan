# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.models.common.linear import Linear


def module_filter_fn(
    mod: nn.Module | Linear.Config, fqn: str, filter_fqns: list[str]
) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.

    Accepts either a built nn.Linear module or a Linear.Config.
    """
    if isinstance(mod, Linear.Config):
        in_features = mod.in_features
        out_features = mod.out_features
    elif isinstance(mod, nn.Linear):
        in_features = mod.weight.shape[1]
        out_features = mod.weight.shape[0]
    else:
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = in_features % 16 == 0 and out_features % 16 == 0

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and not is_filtered_fqn
