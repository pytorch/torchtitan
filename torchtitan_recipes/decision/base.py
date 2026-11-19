# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable components for candidate-distribution decision fine-tuning."""

import math
from abc import ABC, abstractmethod
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
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper, Loss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_CANDIDATES = 2
MAX_CANDIDATES = len(LETTERS)


class DecisionSampleProcessor(SampleProcessor):
    """Turn candidate distributions into sampled next-token decision examples."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length
        self._candidate_token_ids = [
            self._single_token(f" {letter}") for letter in LETTERS
        ]
        self._readout_token_id = self._single_token("\n")

    def _single_token(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Expected {text!r} to encode as one token, got {token_ids}"
            )
        return token_ids[0]

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        candidate_ids = sample["candidate_ids"]
        options = sample["options"]
        target_probs = np.asarray(sample["target_probs"], dtype=np.float64)
        num_candidates = len(candidate_ids)
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"Expected 2 to 26 options, got {num_candidates}")
        if len(options) != num_candidates or target_probs.shape != (num_candidates,):
            raise ValueError(
                "Options, candidate IDs, and target probabilities must align"
            )

        order = np.arange(num_candidates)
        if sample["kind"] != "score":
            rng.shuffle(order)
        ordered_options = [options[index] for index in order]
        ordered_probs = target_probs[order]
        if (
            not np.isfinite(ordered_probs).all()
            or (ordered_probs < 0).any()
            or not np.isclose(ordered_probs.sum(), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError("target_probs must be finite, nonnegative, and sum to one")
        ordered_probs /= ordered_probs.sum()

        target_index = rng.choice(num_candidates, p=ordered_probs)
        option_text = "\n".join(
            f"{LETTERS[index]}) {description}"
            for index, description in enumerate(ordered_options)
        )
        prompt = (
            f"State:\n{sample['state']}\n\n"
            f"Question ({sample['kind']}): {sample['instructions']}\n"
            f"Options:\n{option_text}\nAnswer:"
        )
        prompt_ids = self._tokenizer.encode(prompt, add_bos=False, add_eos=False)
        input_ids = np.asarray([*prompt_ids, self._readout_token_id], dtype=np.int64)
        if len(input_ids) > self._max_context_length:
            return None

        labels = np.full(len(input_ids), IGNORE_INDEX, dtype=np.int64)
        labels[-2] = -num_candidates
        labels[-1] = self._candidate_token_ids[target_index]
        return TextSequence(input_ids=input_ids, labels=labels)


class CandidateDistributionLoss(Loss[torch.Tensor]):
    """Estimate candidate-distribution NLL plus expected multiclass Brier."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        candidate_token_ids: list[int]
        brier_weight: float = 1.0

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.brier_weight) or config.brier_weight < 0:
            raise ValueError("brier_weight must be finite and nonnegative")
        if len(config.candidate_token_ids) != MAX_CANDIDATES:
            raise ValueError("candidate_token_ids must contain IDs for A through Z")
        self.candidate_token_ids = config.candidate_token_ids
        self.brier_weight = config.brier_weight
        self._candidate_ids_by_device: dict[str, torch.Tensor] = {}
        self._candidate_positions_by_device: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        *,
        candidate_counts: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        device_key = str(pred.device)
        candidate_ids = self._candidate_ids_by_device.get(device_key)
        positions = self._candidate_positions_by_device.get(device_key)
        if candidate_ids is None:
            candidate_ids = torch.tensor(self.candidate_token_ids, device=pred.device)
            positions = torch.arange(MAX_CANDIDATES, device=pred.device)
            self._candidate_ids_by_device[device_key] = candidate_ids
            self._candidate_positions_by_device[device_key] = positions
        logits = pred.float().index_select(-1, candidate_ids)
        valid = labels != IGNORE_INDEX
        candidate_counts = candidate_counts.to(device=pred.device, dtype=torch.long)
        candidate_mask = (
            positions < candidate_counts[:, None]
        )
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets = labels[:, None] == candidate_ids[None, :]
        target_indices = targets.to(torch.long).argmax(dim=-1)
        target_is_valid = targets.any(dim=-1) & (target_indices < candidate_counts)
        torch._assert_async(
            (target_is_valid | ~valid).all(),
            "Answer token must be one of the valid candidate slots",
        )

        target_log_probs = log_probs.gather(1, target_indices[:, None]).squeeze(1)
        target_probs = probs.gather(1, target_indices[:, None]).squeeze(1)
        nll = -target_log_probs[valid].sum()
        brier = (probs.square().sum(dim=-1) + 1 - 2 * target_probs)[valid].sum()
        denominator = (
            global_loss_token_counts / 2
            if global_loss_token_counts is not None
            else valid.sum()
        )
        loss = (nll + self.brier_weight * brier) / denominator
        return loss, {
            "decision_nll/mean": nll.detach() / denominator,
            "decision_sampled_brier/mean": brier.detach() / denominator,
        }


class DecisionLoss(ChunkedLossWrapper):
    """Project decision positions and carry option counts through packed labels."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        max_num_supervised: int
        """Maximum number of questions in one packed microbatch."""
        num_chunks: int = 1

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_num_supervised = config.max_num_supervised

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count_markers = (labels <= -MIN_CANDIDATES) & (labels >= -MAX_CANDIDATES)
        torch._assert_async(
            count_markers.sum() <= self.max_num_supervised,
            "more decision questions than max_num_supervised",
        )
        marker_indices = torch.nonzero_static(
            count_markers, size=self.max_num_supervised, fill_value=-1
        ).squeeze(-1)
        valid = marker_indices >= 0
        safe_marker_indices = marker_indices.clamp(min=0)
        answer_indices = safe_marker_indices + 1
        torch._assert_async(
            (~valid | (answer_indices < labels.numel())).all(),
            "decision count marker must be followed by an answer label",
        )
        answer_labels = labels[answer_indices].masked_fill(~valid, IGNORE_INDEX)
        candidate_counts = (-labels[safe_marker_indices]).clamp(
            min=MIN_CANDIDATES, max=MAX_CANDIDATES
        )
        return super().__call__(
            pred[answer_indices],
            answer_labels,
            global_loss_token_counts,
            candidate_counts=candidate_counts,
            **loss_inputs,
        )


class DecisionModel(ABC):
    """Model-family hooks shared by TorchTitan candidate-decision recipes."""

    def __init__(
        self,
        model_size: str,
        *,
        parallelism: ParallelismConfig | None = None,
        brier_weight: float = 1.0,
    ) -> None:
        if not math.isfinite(brier_weight) or brier_weight < 0:
            raise ValueError("brier_weight must be finite and nonnegative")
        self.model_size = model_size
        self.parallelism = parallelism or ParallelismConfig()
        self.brier_weight = brier_weight
        if (
            self.parallelism.tensor_parallel_degree > 1
            or self.parallelism.pipeline_parallel_degree > 1
            or self.parallelism.context_parallel_degree > 1
            or self.parallelism.expert_parallel_degree > 1
        ):
            raise ValueError(
                "This decision recipe currently supports data parallelism only"
            )

    @property
    @abstractmethod
    def hf_assets_path(self) -> str:
        """Return the pretrained HF assets directory for this model family."""

    @abstractmethod
    def build_backbone_config(self, *, seq_len: int) -> tuple[Configurable.Config, int]:
        """Return the TorchTitan backbone config and its maximum context length."""

    def build_trainer_config(
        self,
        *,
        data_path: str,
        max_num_questions: int = 256,
        steps: int = 600,
    ) -> Trainer.Config:
        """Assemble shared candidate-distribution training components."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=self.hf_assets_path)
        candidate_token_ids = [
            tokenizer.encode(f" {letter}", add_bos=False, add_eos=False)
            for letter in LETTERS
        ]
        if any(len(token_ids) != 1 for token_ids in candidate_token_ids):
            raise ValueError(
                "The tokenizer must encode each space-prefixed answer letter as one token"
            )
        candidate_token_ids = [token_ids[0] for token_ids in candidate_token_ids]
        if len(set(candidate_token_ids)) != MAX_CANDIDATES:
            raise ValueError("Answer-letter tokens must be distinct")

        model_config, max_context_length = self.build_backbone_config(seq_len=8192)
        num_tokens_per_microbatch = 4 * max_context_length
        return Trainer.Config(
            loss=DecisionLoss.Config(
                loss_fn=CandidateDistributionLoss.Config(
                    candidate_token_ids=candidate_token_ids,
                    brier_weight=self.brier_weight,
                ),
                max_num_supervised=max_num_questions,
            ),
            hf_assets_path=self.hf_assets_path,
            model=model_config,
            parallelism=self.parallelism,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, weight_decay=0.0)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    decay_ratio=0.9,
                    decay_type="cosine",
                    min_lr_factor=0.1,
                ),
            ),
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
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
                        processor=DecisionSampleProcessor.Config(),
                        post_filters=(lambda sample: sample is not None,),
                    ),
                ),
                max_num_documents=max_num_questions,
                read_options=grain.ReadOptions(num_threads=0),
                num_prefetch_microbatches=4,
            ),
            metrics=MetricsProcessor.Config(log_freq=10),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                last_save_in_hf=True,
                export_dtype="bfloat16",
            ),
        )
