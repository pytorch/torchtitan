# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import version

# Import to register quantization modules.
import torchtitan.components.quantization  # noqa: F401

try:
    __version__ = version("torchtitan")
except Exception as e:
    __version__ = "0.0.0+unknown"
