# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""cutlass-dsl / nccl4py version-compatibility shim.

nccl4py (through 0.3.1) imports ``dsl_user_op`` from
``cutlass.base_dsl._mlir_helpers.op``, but cutlass-dsl 4.6.x relocated that
module to top-level ``cutlass._mlir_helpers.op``. nccl4py's pin still allows
4.6.x, so importing the nccl cute bindings fails with ModuleNotFoundError.

Importing this module aliases the module back to where nccl4py looks for it.
It is a no-op on cutlass-dsl 4.5.x (the dev box) and on hosts without cutlass
(the portable p2p path), so it is always safe to import first.
"""

import sys


def _install() -> None:
    try:
        import cutlass.base_dsl._mlir_helpers.op  # noqa: F401

        return  # already where nccl4py expects it (cutlass-dsl 4.5.x)
    except ModuleNotFoundError:
        pass
    try:
        import cutlass._mlir_helpers as _mlir_helpers
        import cutlass._mlir_helpers.op as _mlir_helpers_op
    except ModuleNotFoundError:
        return  # no cutlass at all -> portable p2p-only host, shim not needed
    sys.modules["cutlass.base_dsl._mlir_helpers"] = _mlir_helpers
    sys.modules["cutlass.base_dsl._mlir_helpers.op"] = _mlir_helpers_op


_install()
