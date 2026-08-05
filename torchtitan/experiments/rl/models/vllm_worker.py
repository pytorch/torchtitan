# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

import torch

from vllm.utils.math_utils import round_up
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner that pads EP+TP batches for TorchTitan's MoE sharding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_ep = self.vllm_config.parallel_config.enable_expert_parallel
        self._torchtitan_num_actual_tokens = (
            torch.full(
                (1,),
                self.max_num_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            if enable_ep and tp_size > 1
            else None
        )

    def _init_model_kwargs(self):
        model_kwargs = super()._init_model_kwargs()
        if self._torchtitan_num_actual_tokens is not None:
            model_kwargs["num_actual_tokens"] = self._torchtitan_num_actual_tokens
        return model_kwargs

    def _preprocess(
        self,
        scheduler_output,
        num_input_tokens: int,
        intermediate_tensors=None,
    ):
        if self._torchtitan_num_actual_tokens is not None:
            self._torchtitan_num_actual_tokens.fill_(
                scheduler_output.total_num_scheduled_tokens
            )
        return super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # This is input padding only. It does not enable vLLM's
        # pass_config.enable_sp Inductor optimization.
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_ep = self.vllm_config.parallel_config.enable_expert_parallel
        if enable_ep and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens


class TorchTitanGPUWorker(GPUWorker):
    """V1 worker that constructs :class:`TorchTitanGPUModelRunner`."""

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError(
                "TorchTitan's vLLM integration requires the V1 model runner"
            )

        # GPUWorker imports its runner class inside init_device and provides no
        # runner factory. Scope the class substitution to that construction.
        original_runner_cls = vllm_gpu_model_runner.GPUModelRunner
        vllm_gpu_model_runner.GPUModelRunner = (  # pyrefly: ignore[bad-assignment]
            TorchTitanGPUModelRunner
        )
        try:
            super().init_device()
        finally:
            vllm_gpu_model_runner.GPUModelRunner = (  # pyrefly: ignore[bad-assignment]
                original_runner_cls
            )
