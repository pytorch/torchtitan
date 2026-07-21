# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Backend selection for the selective gather.

``"p2p"`` sends over ``batch_isend_irecv``: it runs anywhere NCCL/RCCL
point-to-point works and needs no nccl4py/CuTeDSL.
"""

import os
import warnings

SUPPORTED_BACKENDS = ("p2p",)


def select_backend(device=None, group=None) -> str:
    """Return the selective-gather backend for this CP group.

    ``SELECTIVE_GATHER_BACKEND`` overrides the choice. An unsupported value
    warns instead of silently taking no effect.
    """
    forced = os.environ.get("SELECTIVE_GATHER_BACKEND")
    if forced is not None and forced not in SUPPORTED_BACKENDS:
        warnings.warn(
            f"SELECTIVE_GATHER_BACKEND={forced!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_BACKENDS)}); using 'p2p'.",
            stacklevel=2,
        )
    return "p2p"
