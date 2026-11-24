# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 training on GS1-format candidate distributions."""

import math
from dataclasses import dataclass
from typing import Any

import grain.python as grain
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
from torchtitan.components.loss import IGNORE_INDEX, Loss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from .loss import AnswerTokenLoss

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
        readout = self._tokenizer.encode("\n", add_bos=False, add_eos=False)
        if len(readout) != 1:
            raise ValueError(f"Expected a one-token newline, got {readout}")
        self._readout_token_id = readout[0]

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        options = sample["options"]
        probs = np.asarray(sample["target_probs"], dtype=np.float64)
        num_candidates = len(sample["candidate_ids"])
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"Expected 2 to 26 options, got {num_candidates}")
        if len(options) != num_candidates or probs.shape != (num_candidates,):
            raise ValueError(
                "Options, candidate IDs, and target probabilities must align"
            )

        order = np.arange(num_candidates)
        if sample["kind"] != "score":
            rng.shuffle(order)
        probs = probs[order]
        total_prob = probs.sum()
        if (
            not np.isfinite(probs).all()
            or (probs < 0).any()
            or not np.isclose(total_prob, 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError("target_probs must be finite, nonnegative, and sum to one")

        probs /= total_prob
        target_index = rng.choice(num_candidates, p=probs)
        option_text = "\n".join(
            f"{LETTERS[slot]}) {options[index]}" for slot, index in enumerate(order)
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
        # One supervised position per question. The loss decodes the option count
        # and the sampled option from this class ID.
        labels[-1] = (num_candidates - MIN_CANDIDATES) * MAX_CANDIDATES + target_index
        return TextSequence(input_ids=input_ids, labels=labels)


class CandidateDistributionLoss(Loss[torch.Tensor]):
    """Estimate candidate-distribution NLL plus expected multiclass Brier."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        candidate_token_ids: list[int] | None = None
        tokenizer_path: str | None = None
        brier_weight: float = 1.0

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.brier_weight) or config.brier_weight < 0:
            raise ValueError("brier_weight must be finite and nonnegative")
        candidate_token_ids = config.candidate_token_ids
        if candidate_token_ids is None:
            if config.tokenizer_path is None:
                raise ValueError(
                    "A tokenizer path is required to resolve answer tokens"
                )
            tokenizer = HuggingFaceTokenizer(tokenizer_path=config.tokenizer_path)
            encoded = [
                tokenizer.encode(f" {letter}", add_bos=False, add_eos=False)
                for letter in LETTERS
            ]
            if any(len(ids) != 1 for ids in encoded):
                raise ValueError("Each answer letter must encode as one token")
            candidate_token_ids = [ids[0] for ids in encoded]
        if (
            len(candidate_token_ids) != MAX_CANDIDATES
            or len(set(candidate_token_ids)) != MAX_CANDIDATES
        ):
            raise ValueError("Expected distinct token IDs for A through Z")
        self.candidate_token_ids = candidate_token_ids
        self.brier_weight = config.brier_weight
        self._candidates_by_device: dict[
            torch.device, tuple[torch.Tensor, torch.Tensor]
        ] = {}

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
        class_ids = labels.masked_fill(~valid, 0)
        candidate_counts = class_ids // MAX_CANDIDATES + MIN_CANDIDATES
        target_indices = class_ids % MAX_CANDIDATES
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


def qwen3_5_distribution(
    model_size: str = "0.8B",
    *,
    parallelism: ParallelismConfig | None = None,
    brier_weight: float = 1.0,
) -> Trainer.Config:
    """Fine-tune a Qwen3.5 base model on GS1-format candidate distributions."""
    max_num_questions = 256
    hf_assets_path = f"./assets/hf/Qwen3.5-{model_size}-Base"
    data_path = "./assets/hf/gs1-distribution/train.jsonl"
    parallelism = parallelism or ParallelismConfig()
    if (
        parallelism.tensor_parallel_degree > 1
        or parallelism.pipeline_parallel_degree > 1
        or parallelism.context_parallel_degree > 1
        or parallelism.expert_parallel_degree > 1
    ):
        raise ValueError(
            "This decision recipe currently supports data parallelism only"
        )
    model_config = build_model_config(model_size, seq_len=8192, parallelism=parallelism)
    return Trainer.Config(
        loss=AnswerTokenLoss.Config(
            loss_fn=CandidateDistributionLoss.Config(
                tokenizer_path=hf_assets_path,
                brier_weight=brier_weight,
            ),
            max_num_supervised=max_num_questions,
        ),
        hf_assets_path=hf_assets_path,
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, weight_decay=0.0)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                decay_ratio=0.9,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            steps=600,
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
