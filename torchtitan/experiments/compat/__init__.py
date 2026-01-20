# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch compatibility shims for non-nightly versions.

This experimental module provides compatibility between PyTorch nightly and stable releases
by shimming missing modules and functions.

Usage:
    import torchtitan.experiments.compat  # noqa: F401

The shims are automatically installed when this module is imported.
"""

# Import compat to auto-install shims
from . import compat  # noqa: F401
