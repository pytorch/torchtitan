# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 training recipes."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    BlockRematOverride,
)
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
from torchtitan.models.common.attention.gdn import InnerGDN
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config, Qwen35Model
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


def qwen3_5_9b_1m_cp8_ulysses_lean_remat_offload() -> Trainer.Config:
    """Qwen3.5 9B text model at 1M context with YaRN RoPE and CP 8.

    Full attention uses Ulysses CP with FA3. Ulysses copies each of the 4 KV
    heads to 2 of the 8 CP ranks. It needs contiguous token shards, so there is
    no load balancer.

    Lean remat keeps full attention out of recompute and offloads what blocks
    retain. Each block keeps only its input, and each full-attention block also
    its attention kernel's output and LSE; everything else, including the CP
    communication, is recomputed. Both go to CPU (about 38 GiB of pinned memory
    per rank at 1M), except in the last block: only the loss runs between its
    forward and its backward, so its copies could not overlap compute.
    """
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "9B_1M", attn_backend="varlen", parallelism=parallelism
    )
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            # 4096 tokens per chunk on each of the 8 CP ranks keeps one bf16
            # logits chunk (248320-token vocab) at about 2 GiB.
            num_chunks=num_tokens_per_microbatch_per_dp_rank // (8 * 4096),
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*")]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=10,
        ),
        parallelism=parallelism,
        checkpointer=None,
    )
    # CP is set after the base config validates; the transform below adds the
    # CP inner attention and validates again.
    config.parallelism.context_parallel_degree = 8
    config = apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: UlyssesVarlenInnerAttention,
                    InnerGDN: ContextParallelInnerGDN,
                }
            ),
        ],
    )
    assert isinstance(config.model, Qwen35Model.Config)
    for layer in config.model.layers:
        inner_attention = layer.attention.inner_attention
        if isinstance(inner_attention, UlyssesVarlenInnerAttention.Config):
            inner_attention.sdpa_backends = ["FLASH_ATTENTION"]
    kernel_regions = ["attention.inner_attention.kernel*"]
    last_block = f"layers.{len(config.model.layers) - 1}"
    config.activation_remat = ActivationRematConfig(
        save_regions=kernel_regions,
        offload_block_input=True,
        offload_regions=kernel_regions,
        block_overrides=[
            BlockRematOverride(
                blocks=[last_block], offload_block_input=False, offload_regions=[]
            ),
        ],
    )
    # TODO: re-enable once CUDA graph capture works with this PyTorch build.
    config.training.disable_accelerator_graphs = True
    return config
