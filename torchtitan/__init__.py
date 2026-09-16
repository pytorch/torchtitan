# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import version

try:
    __version__ = version("torchtitan")
except Exception:
    __version__ = "0.0.0+unknown"

# Import torch (and thereby the torch_npu backend) before any `spmd_types`
# import. torch_npu 2.14 eagerly initializes its distributed stack during
# `import torch`, which pulls in torch.distributed.fsdp and re-enters
# `spmd_types` while it is still mid-import -> circular-import AttributeError
# (`spmd_types has no attribute 'register_local_autograd_function'`).
import torch  # noqa: E402,F401
