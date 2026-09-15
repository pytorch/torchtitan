# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from torchtitan.experiments.rl.models.gdn_backend import (
    TorchTitanGDNAttentionBackend,
    TorchTitanGDNAttentionMetadataBuilder,
)
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


logger = init_logger(__name__)


class TorchTitanGDNGraphWrapper(BreakableCUDAGraphWrapper):
    """Stage group-owned GDN metadata before upstream graph capture or replay."""

    def __init__(
        self, runnable: Callable[..., Any], runner: "TorchTitanGPUModelRunner"
    ) -> None:
        if runner.compilation_config.cudagraph_num_of_warmups < 1:
            raise ValueError(
                "TorchTitan GDN piecewise capture requires "
                "cudagraph_num_of_warmups >= 1 to prewarm lazy kernel variants."
            )
        super().__init__(runnable, runner.vllm_config)
        self.runner = runner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available():
            return super().__call__(*args, **kwargs)

        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.FULL or (
            context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and context.attn_metadata is not None
        ):
            return super().__call__(*args, **kwargs)

        # Graph-memory profiling can replace these builders before real capture.
        groups = [
            group for cache_group in self.runner.attn_groups for group in cache_group
        ]
        gdn_groups = [
            (group, builder)
            for group in groups
            if isinstance(
                builder := group.get_metadata_builder(),
                TorchTitanGDNAttentionMetadataBuilder,
            )
        ]
        if not gdn_groups:
            return super().__call__(*args, **kwargs)

        live_metadata = context.attn_metadata
        assert live_metadata is None or isinstance(live_metadata, dict)
        if live_metadata is not None and any(
            isinstance(
                metadata := live_metadata[group.layer_names[0]], GDNAttentionMetadata
            )
            and metadata.num_prefills == 0
            and metadata.num_actual_tokens > 0
            for group, _ in gdn_groups
        ):
            # Packed prefill and fused decode have different rounding. FULL
            # decode already bypassed this wrapper; preserve the legacy kernel
            # when a pure-decode batch is dispatched to PIECEWISE instead.
            # This integration is enabled only for DP=1, so changing mode here
            # cannot conflict with cross-DP graph-mode coordination.
            logger.warning_once(
                "Running pure-decode PIECEWISE batches eagerly to preserve GDN "
                "decode numerics. Use FULL_AND_PIECEWISE for captured decode."
            )
            with override_forward_context(
                replace(context, cudagraph_runtime_mode=CUDAGraphMode.NONE)
            ):
                return self.runnable(*args, **kwargs)

        # NONE-mode dummy warmups may have no batch descriptor. The runner
        # always supplies positions, whose last axis is the padded token axis.
        token_capacity = (
            context.batch_descriptor.num_tokens
            if context.batch_descriptor is not None
            else kwargs["positions"].shape[-1]
        )
        attn_metadata: dict[str, object] = (
            {name: None for group in groups for name in group.layer_names}
            if live_metadata is None
            else dict(live_metadata)
        )
        for group, builder in gdn_groups:
            if live_metadata is None:
                packed = builder.stage_dummy(token_capacity=token_capacity)
            else:
                metadata = live_metadata[group.layer_names[0]]
                assert isinstance(metadata, GDNAttentionMetadata)
                packed = builder.stage_packed(metadata, token_capacity=token_capacity)
            for name in group.layer_names:
                attn_metadata[name] = packed

        with override_forward_context(replace(context, attn_metadata=attn_metadata)):
            return super().__call__(*args, **kwargs)


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner with sequence-parallel padding and packed GDN graph staging."""

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        if not any(
            layer.get_attn_backend() is TorchTitanGDNAttentionBackend
            for layer in get_layers_from_vllm_config(
                self.vllm_config, MambaBase
            ).values()
        ):
            return

        assert isinstance(self.model, BreakableCUDAGraphWrapper)
        assert not self.model.entries
        self.model = TorchTitanGDNGraphWrapper(self.model.unwrap(), self)

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_dense_sp = self.compilation_config.pass_config.enable_sp and tp_size > 1
        enable_expert_sp = (
            self.vllm_config.parallel_config.enable_expert_parallel and tp_size > 1
        )
        if enable_dense_sp or enable_expert_sp:
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
