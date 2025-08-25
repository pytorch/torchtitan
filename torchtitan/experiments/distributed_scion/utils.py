# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re


def remove_orig_mod_and_weight_for_p_name(name: str) -> str:
    """
    Remove "._orig_mod", ".weight", and "._checkpoint_wrapped_module" to
    get the clean layer name.
    """
    name = re.sub(r"\._orig_mod", "", name)  # comes from compiled model
    name = re.sub(r"\.weight", "", name)  # param.weight
    name = re.sub(
        r"\._checkpoint_wrapped_module", "", name
    )  # comes from activation checkpointing
    return name
