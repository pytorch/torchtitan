# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 quantization building blocks."""

_mxfp8_linear_import_error: ImportError | None = None

try:
    # The 32x32 swizzled cast kernels are newer than released torchao builds.
    # Keep the package importable for other quantization modes and defer the
    # actionable error to MXFP8LinearConverter construction.
    from .linear import MXFP8Linear
except ImportError as import_error:
    MXFP8Linear = None
    _mxfp8_linear_import_error = import_error


__all__ = ["MXFP8Linear"]
