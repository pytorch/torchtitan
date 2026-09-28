# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float8 quantization building blocks."""

_float8_linear_import_error: ImportError | None = None

try:
    from .linear import Float8Linear
except ImportError as import_error:
    Float8Linear = None
    _float8_linear_import_error = import_error

_float8_experts_import_error: ImportError | None = None

try:
    from .experts import _float8_grouped_linear_cache, _get_float8_grouped_linear_cls
except ImportError as import_error:
    _float8_grouped_linear_cache = {}
    _get_float8_grouped_linear_cls = None
    _float8_experts_import_error = import_error


__all__ = ["Float8Linear"]
