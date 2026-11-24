# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 training recipes."""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
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
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
from torchtitan.models.common.attention.gdn import InnerGDN
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config, Qwen35Model
from torchtitan.trainer import Trainer


_DNA_VOCAB_SIZE = 128


class _FP32LMHead(Linear):
    """LM head with an FP32 weight and an FP32 matmul, as in the Blackberry config.

    ``preserve_dtype_params`` keeps the weight in FP32 under FSDP mixed
    precision; the BF16 hidden state is upcast for the matmul.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.linear(input.float(), weight, bias)


def _qwen3_5_9b_ulysses_lean_remat_offload(
    flavor: str, *, cp: int, max_num_documents: int
) -> Trainer.Config:
    """Qwen3.5 9B text model at long context with YaRN RoPE and Ulysses CP.

    The vocabulary is the 128-token DNA vocabulary of the Blackberry config
    (evo-design/blackberry, configs/torchtitan/from_scratch_1m.json), not
    Qwen's 248320 tokens. The logits are small, so the loss needs no chunking.
    As in that config, the LM head runs in FP32.

    Full attention uses Ulysses CP with FA3. Ulysses copies each of the 4 KV
    heads to ``cp // 4`` CP ranks. It needs contiguous token shards, so there
    is no load balancer.

    Lean remat keeps full attention out of recompute and offloads what blocks
    retain. Each block keeps only its input, and each full-attention block also
    its attention kernel's output and LSE; everything else, including the CP
    communication, is recomputed. Both go to CPU (about 38 GiB of pinned memory
    per rank at 128K tokens per rank), except in the last block: only the loss
    runs between its forward and its backward, so its copies could not overlap
    compute.

    ``max_num_documents`` caps document pieces per row (about 9 MiB of GDN
    state per slot). The recipes size it for the OpenGenome3 DNA sample, whose
    shortest document has 131351 tokens, so an N-token row holds at most
    ``(N - 1) // 131351 + 2`` pieces and never pads.
    """
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        flavor, attn_backend="varlen", parallelism=parallelism
    )
    model_config.vocab_size = _DNA_VOCAB_SIZE
    model_config.tok_embeddings.num_embeddings = _DNA_VOCAB_SIZE
    model_config.lm_head = _FP32LMHead.Config(
        **{
            f.name: getattr(model_config.lm_head, f.name)
            for f in fields(model_config.lm_head)
            if f.init
        }
        | dict(
            out_features=_DNA_VOCAB_SIZE,
            preserve_dtype_params=("weight",),
            inplace_wgrad_accum=False,
        )
    )
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    config = Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=max_num_documents,
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
    config.parallelism.context_parallel_degree = cp
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
    # The fused SwiGLU's backward overwrites w13's output with its gradient:
    # one [T, 2, F] tensor less at the MLP backward peak (12 GiB at 256K tokens
    # per rank). Lean remat recomputes the MLP, so no other op saves w13's output.
    config.override.imports = ["torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"]
    return config


def qwen3_5_9b_1m_cp8_ulysses_lean_remat_offload() -> Trainer.Config:
    """1M context, CP 8 (128K tokens per rank)."""
    # max_num_documents: (2**20 - 1) // 131351 + 2 = 9
    return _qwen3_5_9b_ulysses_lean_remat_offload("9B_1M", cp=8, max_num_documents=9)


def qwen3_5_9b_2m_cp16_ulysses_lean_remat_offload() -> Trainer.Config:
    """2M context, CP 16 (128K tokens per rank)."""
    # max_num_documents: (2**21 - 1) // 131351 + 2 = 17
    return _qwen3_5_9b_ulysses_lean_remat_offload("9B_2M", cp=16, max_num_documents=17)


def qwen3_5_9b_4m_cp16_ulysses_lean_remat_offload() -> Trainer.Config:
    """4M context, CP 16 (256K tokens per rank; Ulysses allows at most CP 16)."""
    # max_num_documents: (2**22 - 1) // 131351 + 2 = 33
    return _qwen3_5_9b_ulysses_lean_remat_offload("9B_4M", cp=16, max_num_documents=33)
