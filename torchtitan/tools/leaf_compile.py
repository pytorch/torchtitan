# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile small pure-tensor functions on their own ("leaf compile").

Why this exists: compiling whole transformer blocks with ``torch.compile``
fused nothing on DeepSeek-V4 flash -- Dynamo abandoned every block frame at an
SPMD typecheck context manager / the eager DSA block-mask build, so the block
body silently ran eager (profile-verified, four attempts). The math that
dominated the eager elementwise bucket lives in a handful of functions that
contain nothing but tensor ops. Compiling those functions individually, with
``fullgraph=True`` so a graph break is an error rather than a silent fallback,
fuses each into a few kernels. On the flash recipe the hyper-connection
functions alone were worth +24 % (99.5 -> 123.3 TFLOP/s, 8 nodes).

Groups let a run ablate one site without a code change:

    TORCHTITAN_LEAF_COMPILE=all            (default) compile every leaf
    TORCHTITAN_LEAF_COMPILE=0              compile none (eager reference)
    TORCHTITAN_LEAF_COMPILE=hc,attn        only the named groups

``HC_COMPILE=0`` (the knob the first measurement used) still disables the
``hc`` group.
"""

import os
from collections.abc import Callable

import torch

_SPEC = os.environ.get("TORCHTITAN_LEAF_COMPILE", "all").strip().lower()


def leaf_compile_enabled(group: str) -> bool:
    if group == "hc" and os.environ.get("HC_COMPILE", "1") == "0":
        return False
    if _SPEC in ("", "0", "none", "off", "false"):
        return False
    if _SPEC in ("all", "1", "true"):
        return True
    return group in {g.strip() for g in _SPEC.split(",")}


def leaf_compile(
    fn: Callable | None = None, *, group: str, dynamic: bool = False
) -> Callable:
    """Decorator: ``torch.compile(fn, fullgraph=True)`` if ``group`` is enabled.

    ``dynamic=False`` for sites whose shapes are fixed per rank (attention,
    hyper-connections); ``dynamic=True`` for sites whose leading dim varies
    step to step (the routed-token count in MoE experts), which would
    otherwise recompile every step.
    """

    def deco(f: Callable) -> Callable:
        if not leaf_compile_enabled(group):
            return f
        return torch.compile(f, fullgraph=True, dynamic=dynamic)

    return deco if fn is None else deco(fn)
