# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's vLLM worker for Intel XPU.

Kept apart from ``vllm_worker`` so CUDA installs never import vLLM's XPU worker.
vLLM's CUDA ``Worker.init_device`` rejects any non-CUDA device, so XPU needs its
own subclass of vLLM's ``XPUWorker`` carrying the same customizations.
"""

from torchtitan.rl.model.vllm_worker import (
    TorchTitanGPUModelRunner,
    TorchTitanWeightsOnlyMemoryPoolMixin,
)
from vllm.v1.worker.xpu_model_runner import XPUModelRunner
from vllm.v1.worker.xpu_worker import XPUWorker


class TorchTitanXPUModelRunner(TorchTitanGPUModelRunner, XPUModelRunner):
    """:class:`TorchTitanGPUModelRunner` overrides on top of vLLM's XPU runner.

    MRO: TorchTitanGPUModelRunner -> XPUModelRunner -> GPUModelRunner, so the
    TorchTitan overrides win and XPUModelRunner's constructor still runs.
    """


class TorchTitanXPUWorker(TorchTitanWeightsOnlyMemoryPoolMixin, XPUWorker):
    """V1 XPU worker that constructs :class:`TorchTitanXPUModelRunner`."""

    def _make_model_runner(self):
        # XPUWorker exposes a runner factory, so no class substitution is needed.
        if self.use_v2_model_runner:
            raise ValueError(
                "TorchTitan's vLLM integration requires the V1 model runner"
            )
        return TorchTitanXPUModelRunner(self.vllm_config, self.device)
