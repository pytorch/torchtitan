# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed-decision fine-tuning of Qwen3-1.7B-Base with a letter-slot readout.

Trains on ``prepare_data.py`` prompts, supervising only the answer-letter
token that ``evaluate.py`` reads. Each row is packed as its own document, and
the default ``varlen`` attention keeps rows from attending to each other. See
README.md for commands and results.
"""

from dataclasses import dataclass
from typing import Any

import grain
import numpy as np
import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.dataset import SampleProcessor, TextSequence
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    IGNORE_INDEX,
)
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ContextParallelTransform,
    transform_model_config_,
)
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


class DecisionLetterProcessor(SampleProcessor):
    """Tokenize ``prompt + answer`` and supervise only the answer token."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length

    def __call__(
        self, sample: dict[str, str], rng: np.random.Generator
    ) -> TextSequence:
        del rng
        prompt = self._tokenizer.encode(sample["prompt"], add_bos=False, add_eos=False)
        answer = self._tokenizer.encode(sample["answer"], add_bos=False, add_eos=False)
        if len(answer) != 1:
            raise ValueError(f"Answer must be one token, got {sample['answer']!r}")
        # Packing would split an over-length row into separate documents, and
        # the answer token would be supervised without the start of its prompt.
        if len(prompt) > self._max_context_length:
            raise ValueError(
                f"Decision row has {len(prompt)} tokens, more than "
                f"max_context_length={self._max_context_length}"
            )
        labels = np.full(len(prompt), IGNORE_INDEX, dtype=np.int64)
        labels[-1] = answer[0]
        return TextSequence(input_ids=np.asarray(prompt, dtype=np.int64), labels=labels)


class AnswerTokenLoss(ChunkedLossWrapper):
    """Apply lm_head and the loss only at supervised positions.

    Gathering a fixed number of positions keeps shapes static for CUDA graphs
    and skips lm_head and cross-entropy over the ~99% of positions whose labels
    are ``IGNORE_INDEX``. ``DecisionLetterProcessor`` supervises one token per
    row and packing caps documents per microbatch at ``max_num_documents``, so
    setting ``max_num_supervised = max_num_documents`` is enough. An on-device
    assert catches a violation instead of silently dropping supervised tokens.

    The gathered rows go through ``ChunkedLossWrapper.__call__``, which applies
    lm_head and the loss with its FSDP and loss-parallel handling.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        max_num_supervised: int
        """Upper bound on supervised positions per microbatch."""
        num_chunks: int = 1
        """The gathered rows are few, so they are not chunked further."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.max_num_supervised = config.max_num_supervised

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        is_supervised_T = labels != IGNORE_INDEX
        torch._assert_async(
            is_supervised_T.sum() <= self.max_num_supervised,
            "more supervised positions than max_num_supervised",
        )
        # Unused slots hold -1: they gather the last row and get ignored labels.
        index_S = torch.nonzero_static(
            is_supervised_T, size=self.max_num_supervised, fill_value=-1
        ).squeeze(-1)
        labels_S = labels[index_S].masked_fill(index_S < 0, IGNORE_INDEX)
        return super().__call__(
            pred[index_S], labels_S, global_loss_token_counts, **loss_inputs
        )


def qwen3_1_7b_decision(parallelism: ParallelismConfig | None = None) -> Trainer.Config:
    """FSDP over all GPUs by default; ``parallelism`` adds TP (with loss
    parallel) or CP (Ulysses varlen attention). PP is unsupported: Qwen3-1.7B ties
    its embedding and output weights."""
    parallelism = parallelism or ParallelismConfig()
    model_config = build_model_config("1.7B", seq_len=2048, parallelism=parallelism)
    if parallelism.context_parallel_degree > 1:
        cp_transform = ContextParallelTransform(
            inner_attention_map={VarlenInnerAttention: UlyssesVarlenInnerAttention}
        )
        # Rewrites the attention configs in place.
        transform_model_config_(model_config, [cp_transform])
    # Decision rows average ~180 tokens, so a 32K-token microbatch holds ~180 of
    # them. Varlen CUDA graphs need this bound, and cuDNN's varlen attention
    # backward cost grows with it.
    max_num_documents = 256
    return Trainer.Config(
        loss=AnswerTokenLoss.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
            max_num_supervised=max_num_documents,
        ),
        hf_assets_path="./assets/hf/Qwen3-1.7B-Base",
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, weight_decay=0.0)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=20,
                decay_ratio=0.9,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=16 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=600,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "./assets/hf/bev-decision-150K/train.jsonl"
                        },
                    ),
                    processor=DecisionLetterProcessor.Config(),
                ),
            ),
            max_num_documents=max_num_documents,
            # Per-sample work is Python-heavy (tokenization); grain's default
            # 16 reader threads contend on the GIL and starve the GPU.
            read_options=grain.ReadOptions(num_threads=0),
            num_prefetch_microbatches=16,
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        ),
        # Without recompute, the 2-GPU FSDP run peaks at ~88 GiB per GB300;
        # enable activation_remat on smaller GPUs.
        activation_remat=None,
    )
