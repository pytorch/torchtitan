# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantization building blocks."""

from .utils import nvfp4_bf16_tail_fqns


_nvfp4_linear_import_error: ImportError | None = None

try:
    from .linear import (
        _HARDCODED_SIGN_VECTOR as _LINEAR_HARDCODED_SIGN_VECTOR,
        NVFP4Linear,
    )

    _HARDCODED_SIGN_VECTOR = _LINEAR_HARDCODED_SIGN_VECTOR
except ImportError as import_error:
    NVFP4Linear = None
    _nvfp4_linear_import_error = import_error


__all__ = ["NVFP4Linear", "nvfp4_bf16_tail_fqns"]
