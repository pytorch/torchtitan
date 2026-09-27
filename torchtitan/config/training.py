# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    num_tokens_per_microbatch_per_dp_rank: int = 16384
    """
    Number of input-token slots processed per data-parallel rank in one model
    forward, before context or tensor parallel sharding.
    """

    num_tokens_per_train_step: int = -1
    """
    Global number of input-token slots across data-parallel ranks, pipeline
    microbatches, and gradient accumulation steps. Defaults to
    `training.num_tokens_per_microbatch_per_dp_rank * num_pp_microbatches *
    data-parallel degree`.
    """

    max_context_length: int = 2048
    """Maximum logical context length used for training."""

    def __post_init__(self) -> None:
        if self.num_tokens_per_microbatch_per_dp_rank <= 0:
            raise ValueError(
                "num_tokens_per_microbatch_per_dp_rank must be greater than 0."
            )
        if self.num_tokens_per_train_step != -1 and self.num_tokens_per_train_step <= 0:
            raise ValueError("num_tokens_per_train_step must be -1 or greater than 0.")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0.")
        if self.max_norm < 0:
            raise ValueError("max_norm must be greater than or equal to 0.")

    max_norm: float | int = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """How many train steps to run"""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    disable_cuda_graphs: bool = False
    """
    Disable CUDA graph capture and replay for the forward+backward step. CUDA
    graphs require fixed-shape inputs and no CPU<->GPU synchronization during
    the captured region. Expert parallelism is supported only with HybridEP
    when ``non_blocking_capacity_factor`` is set. Other EP backends synchronize
    with the host during dispatch. For pipeline parallelism, TorchTitan
    configures the schedule-derived directed-edge process groups required by
    looped and split-backward schedule replay. CUDA graphs are independent of
    ``torch.compile(mode="reduce-overhead")``, which performs its own CUDA graph
    capture.
    """

    dtype: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype for training. In contrast to mixed precision training, setting training_dtype=bfloat16 will
    put all parameters, gradients, and optimizer states in bfloat16, without an extra copy of fp32 weights.
    In the case of full bf16 training, RoPE calculations and logits will still be in fp32.
    """

    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via fully_shard or torch.autocast.
    This feature takes effect via fully_shard when data_parallel_shard_degree > 1 or
    context_parallel_degree > 1; it takes effect via torch.autocast when data_replicate_degree >= 1
    and no other parallelism is enabled, i.e. under DDP or single-device training.
    """

    mixed_precision_reduce: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    gc_debug: bool = False
    """
    Enable GC debugging mode. This will perform gc.collect() at every step to
    detect if there is a reference cycle that includes a CUDA Tensor.
    Note that you may want to lower the training steps to avoid generating too
    many temporary files.
    """
