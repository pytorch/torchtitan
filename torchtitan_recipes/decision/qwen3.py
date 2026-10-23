# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed-decision fine-tuning of Qwen3-1.7B-Base with a letter-slot readout.

Each sample is one (state, question) prompt rendered by ``prepare_data.py``
that ends in ``Answer:``. Only the single answer-letter token is supervised, so
the model learns a distribution over the option letters that ``evaluate.py``
reads in one forward pass. Packing keeps rows as separate documents, so
questions about the same state never attend to each other.

Run with
``--module torchtitan_recipes.decision.qwen3 --config qwen3_1_7b_decision``.
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
    ) -> TextSequence | None:
        del rng
        prompt = self._tokenizer.encode(sample["prompt"], add_bos=False, add_eos=False)
        answer = self._tokenizer.encode(sample["answer"], add_bos=False, add_eos=False)
        if len(answer) != 1:
            raise ValueError(f"Answer must be one token, got {sample['answer']!r}")
        tokens = np.asarray(prompt + answer, dtype=np.int64)
        if len(tokens) - 1 > self._max_context_length:
            return None
        labels = np.full(len(tokens) - 1, IGNORE_INDEX, dtype=np.int64)
        labels[-1] = tokens[-1]
        return TextSequence(input_ids=tokens[:-1], labels=labels)


class AnswerTokenLoss(ChunkedLossWrapper):
    """Apply lm_head and the loss only at supervised positions.

    Gathering a fixed number of positions keeps shapes static for CUDA graphs
    and skips lm_head and cross-entropy over the ~99% of positions whose labels
    are ``IGNORE_INDEX``. The bound holds because ``DecisionLetterProcessor``
    supervises exactly one token per document and drops (never splits)
    over-length documents, and packing caps documents per microbatch at
    ``max_num_documents``. An on-device assert catches a violation instead of
    silently dropping supervised tokens.

    TODO: Subclasses ``ChunkedLossWrapper`` only so the trainer runs the decoder
    with ``_skip_lm_head`` and hands this loss the ``lm_head``; a core hook for
    losses that apply lm_head themselves would remove this.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        max_num_supervised: int
        """Upper bound on supervised positions per microbatch."""

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
        assert self.lm_head is not None, "Set lm_head before calling AnswerTokenLoss"
        hidden_TD = pred.reshape(-1, pred.shape[-1])
        labels_T = labels.reshape(-1)
        is_supervised_T = labels_T != IGNORE_INDEX
        torch._assert_async(
            is_supervised_T.sum() <= self.max_num_supervised,
            "more supervised positions than max_num_supervised",
        )
        index_S = torch.nonzero_static(
            is_supervised_T, size=self.max_num_supervised, fill_value=-1
        ).squeeze(-1)
        is_valid_S = index_S >= 0
        index_S = index_S.clamp(min=0)
        labels_S = torch.where(is_valid_S, labels_T[index_S], IGNORE_INDEX)
        logits_SV = self.lm_head(hidden_TD[index_S])
        return self.loss_fn(logits_SV, labels_S, global_loss_token_counts)


def qwen3_1_7b_decision() -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config("1.7B", seq_len=2048, parallelism=parallelism)
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
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
            max_num_documents=max_num_documents,
            # Per-sample work is Python-heavy (tokenization); grain's default
            # 16 reader threads contend on the GIL and starve the GPU.
            read_options=grain.ReadOptions(num_threads=0, prefetch_buffer_size=500),
            num_prefetch_microbatches=16,
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        ),
        # Activations of a 1.7B model at 32K tokens fit without recompute.
        activation_remat=None,
    )
