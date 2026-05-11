# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive scalar formatting for console metric display."""

from __future__ import annotations

from typing import Any


def fmt_metric_value(value: Any) -> str:
    """Format a value for console display.

    Examples:

        fmt_metric_value(3.67060)   -> '3.67'
        fmt_metric_value(0.00123)   -> '0.0012'
        fmt_metric_value(1234.5)    -> '1234.5'
        fmt_metric_value(2e-6)      -> '2.0e-06'
        fmt_metric_value(True)      -> 'True'
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if value == 0:
        return "0"
    if isinstance(value, int) or abs(value) >= 100:
        return f"{value:.1f}"

    frac = abs(value) - int(abs(value))
    if frac == 0:
        return f"{value:.1f}"

    # Walk decimal digits until we've seen two non-zero ones, capped at 5.
    n_decimals, n_nonzero = 0, 0
    temp = frac
    while n_decimals < 5 and n_nonzero < 2:
        n_decimals += 1
        temp *= 10
        if int(temp) % 10 != 0 or n_nonzero > 0:
            n_nonzero += 1
    if n_nonzero == 0:
        return f"{value:.1e}"
    return f"{value:.{max(n_decimals, 2)}f}"
