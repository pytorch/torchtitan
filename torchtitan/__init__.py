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

# torch_npu has no Inductor backend for the flex attention / block-mask
# machinery. torchtitan's own attention helpers already fall back to eager on
# 'npu', but the context-parallel attention shipped inside torch compiles
# `create_block_mask` via torch.compile (dynamic=False, fullgraph=True), which
# the torch_npu inductor cannot lower (e.g. `aten.sum.dim` with `strict_sum`).
# Force the same eager fallback on the CP path when running on Ascend NPU.
if torch._utils._get_available_device_type() == "npu":
    try:
        import torch.distributed.tensor.experimental._context_parallel._attention as _cp_attn  # noqa: E402
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402

        _cp_attn._compiled_create_block_mask = create_block_mask
    except Exception:  # pragma: no cover - best-effort compat hook
        pass
