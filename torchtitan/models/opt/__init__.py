# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# <model name> is licensed under the <license name>,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.opt.model import ModelArgs, OPT
from torchtitan.models.opt.utils import download_opt_weights, export_opt_weights

__all__ = ["OPT", "download_opt_weights", "export_opt_weights"]

opt_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=8),
    "125M": ModelArgs(dim=768, n_layers=12, n_heads=12),
    "1.3B": ModelArgs(dim=2048, n_layers=24, n_heads=32)
}