# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed-decision fine-tuning of Qwen3 and Qwen3.5 with a letter-slot readout.

Trains on ``prepare_data.py`` prompts, supervising only the answer-letter
token that ``evaluate.py`` reads. Each row is packed as its own document, and
the default ``varlen`` attention keeps rows from attending to each other. See
README.md for commands and results.
"""

import math
from dataclasses import dataclass
from typing import Any

import grain
import numpy as np
import torch
import torch.nn.functional as F

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
    Loss,
)
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import BaseTokenizer
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
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from .prepare_data import LETTERS, MIN_CANDIDATES

MAX_CANDIDATES = len(LETTERS)
# Reserve negative IDs below IGNORE_INDEX, outside the CE vocabulary.
CANDIDATE_LABEL_START = IGNORE_INDEX - 1


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


class DecisionLetterProcessor(SampleProcessor):
    """Tokenize ``prompt + answer`` and supervise only the answer token."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence:
        prompt = self._tokenizer.encode(sample["prompt"], add_bos=False, add_eos=False)
        probs = np.asarray(sample["target_probs"], dtype=np.float64)
        count = len(sample["keys"])
        if not MIN_CANDIDATES <= count <= MAX_CANDIDATES or probs.shape != (count,):
            raise ValueError(
                "Expected 2 to 26 candidates with aligned target probabilities"
            )
        slot = int(rng.choice(count, p=probs))
        label = self._label(count, slot)
        # Packing would split an over-length row into separate documents, and
        # the answer token would be supervised without the start of its prompt.
        if len(prompt) > self._max_context_length:
            raise ValueError(
                f"Decision row has {len(prompt)} tokens, more than "
                f"max_context_length={self._max_context_length}"
            )
        labels = np.full(len(prompt), IGNORE_INDEX, dtype=np.int64)
        labels[-1] = label
        return TextSequence(input_ids=np.asarray(prompt, dtype=np.int64), labels=labels)

    def _label(self, count: int, slot: int) -> int:
        answer = " " + LETTERS[slot]
        ids = self._tokenizer.encode(answer, add_bos=False, add_eos=False)
        if len(ids) != 1:
            raise ValueError(f"Answer must be one token, got {answer!r}")
        return ids[0]


class CandidateLetterProcessor(DecisionLetterProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(DecisionLetterProcessor.Config):
        pass

    def _label(self, count: int, slot: int) -> int:
        return CANDIDATE_LABEL_START - (
            (count - MIN_CANDIDATES) * MAX_CANDIDATES + slot
        )


class CandidateDistributionLoss(Loss[torch.Tensor]):
    """Estimate candidate-distribution NLL plus expected multiclass Brier."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        brier_weight: float = 1.0

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.brier_weight) or config.brier_weight < 0:
            raise ValueError("brier_weight must be finite and nonnegative")
        self.brier_weight = config.brier_weight
        self._candidates_by_device: dict[
            torch.device, tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def set_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        encoded = [
            tokenizer.encode(f" {letter}", add_bos=False, add_eos=False)
            for letter in LETTERS
        ]
        if (
            any(len(ids) != 1 for ids in encoded)
            or len({ids[0] for ids in encoded}) != MAX_CANDIDATES
        ):
            raise ValueError("Expected distinct single-token IDs for A through Z")
        self.candidate_token_ids = [ids[0] for ids in encoded]

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        candidates = self._candidates_by_device.get(pred.device)
        if candidates is None:
            candidates = (
                torch.tensor(self.candidate_token_ids, device=pred.device),
                torch.arange(MAX_CANDIDATES, device=pred.device),
            )
            self._candidates_by_device[pred.device] = candidates
        candidate_ids, positions = candidates
        logits = pred.index_select(-1, candidate_ids).float()
        valid = labels != IGNORE_INDEX
        class_ids = (CANDIDATE_LABEL_START - labels).masked_fill(~valid, 0)
        candidate_counts = class_ids // MAX_CANDIDATES + MIN_CANDIDATES
        target_indices = class_ids % MAX_CANDIDATES
        torch._assert_async(
            (
                (class_ids >= 0)
                & (candidate_counts <= MAX_CANDIDATES)
                & (target_indices < candidate_counts)
            ).all(),
            "Candidate loss requires encoded option counts and slots",
        )
        logits = logits.masked_fill(
            positions >= candidate_counts[:, None], float("-inf")
        )
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target_indices[:, None]).squeeze(1)
        target_probs = probs.gather(1, target_indices[:, None]).squeeze(1)
        nll = -target_log_probs.masked_fill(~valid, 0).sum()
        brier = (
            (probs.square().sum(dim=-1) + 1 - 2 * target_probs)
            .masked_fill(~valid, 0)
            .sum()
        )
        denominator = (
            global_loss_token_counts
            if global_loss_token_counts is not None
            else valid.sum()
        )
        loss = (nll + self.brier_weight * brier) / denominator
        return loss, {
            "decision_nll/mean": nll.detach() / denominator,
            "decision_sampled_brier/mean": brier.detach() / denominator,
        }


class DecisionTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        loss = self.engine.loss_fn
        if isinstance(loss, AnswerTokenLoss) and isinstance(
            loss.loss_fn, CandidateDistributionLoss
        ):
            try:
                loss.loss_fn.set_tokenizer(self.tokenizer)
            except ValueError:
                self.close()
                raise


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
    return _decision_config(
        model_config=model_config,
        parallelism=parallelism,
        hf_assets_path="./assets/hf/Qwen3-1.7B-Base",
        data_path="./assets/hf/bev-decision-150K/train.jsonl",
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config)
        ),
        batch_contexts=16,
        prefetch_microbatches=16,
        warmup_steps=20,
        steps=600,
    )


def qwen3_5_distribution(
    model_size: str = "0.8B",
    *,
    parallelism: ParallelismConfig | None = None,
    brier_weight: float = 1.0,
    loss_fn: Loss.Config | None = None,
) -> Trainer.Config:
    """Train on typed decisions with candidate NLL and Brier or full-vocabulary CE."""
    parallelism = parallelism or ParallelismConfig()
    # Candidate indexing assumes a local vocabulary, and the hybrid backbone
    # uses a packed context per rank. Use the tested data-parallel path here.
    if (
        max(
            parallelism.tensor_parallel_degree,
            parallelism.pipeline_parallel_degree,
            parallelism.context_parallel_degree,
            parallelism.expert_parallel_degree,
        )
        > 1
    ):
        raise ValueError("The distribution recipe supports data parallelism only")
    model_config = build_qwen3_5_config(
        model_size, seq_len=8192, parallelism=parallelism
    )
    assets = f"./assets/hf/Qwen3.5-{model_size}-Base"
    loss_fn = loss_fn or CandidateDistributionLoss.Config(brier_weight=brier_weight)
    return _decision_config(
        model_config=model_config,
        parallelism=parallelism,
        hf_assets_path=assets,
        data_path="./assets/hf/bev-decision-150K/train.jsonl",
        loss_fn=loss_fn,
        batch_contexts=2,
        prefetch_microbatches=2,
        warmup_steps=0,
        steps=120,
    )


def _decision_config(
    *,
    model_config,
    parallelism,
    hf_assets_path,
    data_path,
    loss_fn,
    batch_contexts,
    prefetch_microbatches,
    warmup_steps,
    steps,
):
    max_num_documents = 256
    return DecisionTrainer.Config(
        loss=AnswerTokenLoss.Config(
            loss_fn=loss_fn,
            max_num_supervised=max_num_documents,
        ),
        hf_assets_path=hf_assets_path,
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, weight_decay=0.0)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=warmup_steps,
                decay_ratio=0.9,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=batch_contexts
            * model_config.max_context_length,
            steps=steps,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": data_path},
                    ),
                    processor=(
                        CandidateLetterProcessor.Config()
                        if isinstance(loss_fn, CandidateDistributionLoss.Config)
                        else DecisionLetterProcessor.Config()
                    ),
                ),
            ),
            max_num_documents=max_num_documents,
            # Per-sample work is Python-heavy (tokenization); grain's default
            # 16 reader threads contend on the GIL and starve the GPU.
            read_options=grain.ReadOptions(num_threads=0),
            num_prefetch_microbatches=prefetch_microbatches,
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
