# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.experiments.inkling.state_dict_adapter import export_merged_checkpoint
from torchtitan.tools import filesystem
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)


def _corpus_path(dataloader: BaseDataLoader.Config) -> Path:
    if isinstance(dataloader, GrainDataLoader.Config):
        if (
            dataloader.shuffle
            or dataloader.balancer is not None
            or dataloader.max_num_documents != 1
        ):
            raise ValueError("Inkling requires one document per row in corpus order")
        packing = dataloader.dataset
        if isinstance(packing, ConcatThenSplitPackingConfig) and isinstance(
            packing.dataset, SingleDatasetConfig
        ):
            dataset = packing.dataset
            if (
                dataset.pre_filters
                or dataset.post_filters
                or dataset.processor != TextProcessor.Config()
            ):
                raise ValueError(
                    "Inkling requires unfiltered documents with the default text processor"
                )
            source = dataset.source
            if (
                isinstance(source, HuggingFaceRandomAccessSource.Config)
                and source.path == "text"
                and source.split == "train"
                and source.load_dataset_kwargs.keys() == {"data_files"}
            ):
                path = source.load_dataset_kwargs["data_files"]
                if isinstance(path, str):
                    if "::" in path or any(char in path for char in "*?["):
                        raise ValueError(
                            f"Corpus path must not contain glob or chained path syntax: {path}"
                        )
                    return Path(path)
    raise ValueError("Inkling requires one local text corpus for checked resume")


def _sha256(file: Path) -> str:
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class InklingTrainer(Trainer):
    """Check the run inputs before native resume and export the merged HF model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.create_seed_checkpoint:
                raise ValueError("Inkling does not support seed checkpoints")
            if self.override.imports:
                raise ValueError(
                    "Inkling requires checked resume inputs to be set in the recipe, not --override"
                )
            if self.training.steps < 1:
                raise ValueError("training.steps must be positive")
            if (
                self.training.num_tokens_per_microbatch_per_dp_rank
                != self.model.max_context_length
            ):
                raise ValueError("Inkling requires one context per microbatch")
            source = Path(self.hf_assets_path).resolve()
            if (
                self.checkpointer is None
                or not self.checkpointer.initial_load_in_hf
                or self.checkpointer.initial_load_path is None
                or Path(self.checkpointer.initial_load_path).resolve() != source
            ):
                raise ValueError(
                    "Inkling requires the same HF source for weights and tokenizer"
                )
            if filesystem.is_remote(self.dump_folder) or filesystem.is_remote(
                self.checkpointer.folder
            ):
                raise ValueError(
                    "Inkling requires local training and checkpoint output"
                )
            run = Path(self.dump_folder).resolve()
            if run.is_relative_to(source):
                raise ValueError(
                    "The training output must be outside the source checkpoint"
                )
            native_checkpoint = (run / self.checkpointer.folder).resolve()
            if not native_checkpoint.is_relative_to(run):
                raise ValueError(
                    "The native checkpoint must be inside the training output"
                )
            if native_checkpoint.is_relative_to(source):
                raise ValueError(
                    "The native checkpoint must be outside the source checkpoint"
                )

    def __init__(self, config: Config) -> None:
        self._source = Path(config.hf_assets_path).resolve()
        self._run = Path(config.dump_folder).resolve()
        self._destination = self._run / f"merged-step-{config.training.steps}"
        if self._destination.exists():
            raise ValueError(f"Merged checkpoint already exists: {self._destination}")
        assert config.checkpointer is not None
        self._verify_inputs(
            config, allow_create=not (self._run / config.checkpointer.folder).exists()
        )
        super().__init__(config)

    def _verify_inputs(self, config: Trainer.Config, *, allow_create: bool) -> None:
        manifest_path = self._run / "run-manifest.json"
        exists = manifest_path.exists()
        if not exists and not allow_create:
            raise ValueError(
                f"Cannot resume or export: missing run manifest in {self._run}"
            )
        corpus = _corpus_path(config.dataloader)
        if not corpus.is_file():
            raise ValueError(f"Corpus must be a local text file: {corpus}")
        source_contents = {
            str(file.relative_to(self._source)): _sha256(file)
            for file in self._source.rglob("*")
            if file.is_file()
        }
        if not source_contents:
            raise ValueError(f"Empty or missing source checkpoint: {self._source}")
        contents = {
            "checkpoint": source_contents,
            "corpus": _sha256(corpus),
            "seq_len": config.model.max_context_length,
        }
        if exists:
            saved = json.loads(manifest_path.read_bytes())
            for key, label in (
                ("checkpoint", "checkpoint contents"),
                ("corpus", "corpus contents"),
                ("seq_len", "seq_len"),
            ):
                if saved.get(key) != contents[key]:
                    raise ValueError(f"Run {self._run} is bound to different {label}")
        else:
            self._run.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("x") as stream:
                json.dump(contents, stream, indent=2, sort_keys=True)
                stream.write("\n")

    def train(self) -> None:
        # Match the published reference: BF16 inputs, no reduced-precision
        # intermediate reductions. Leave the caller's split-K policy intact.
        matmul = torch.backends.cuda.matmul
        previous = matmul.allow_bf16_reduced_precision_reduction
        split_k = matmul.allow_bf16_reduced_precision_reduction_split_k
        try:
            matmul.allow_bf16_reduced_precision_reduction = (False, split_k)
            super().train()
        finally:
            matmul.allow_bf16_reduced_precision_reduction = (previous, split_k)
        steps = self.config.training.steps
        if self.engine.num_completed_steps != steps:
            raise ValueError(
                f"Requested step {steps}, but the trainer finished at "
                f"step {self.engine.num_completed_steps}"
            )
        (model,) = self.engine.model_parts
        # LoRA B starts at zero. Report the learned projections, not just a loss.
        b_norms = {
            name: param.detach().float().norm().item()
            for name, param in model.named_parameters()
            if param.requires_grad and name.endswith(".lora_b.weight")
        }
        if not all(map(math.isfinite, b_norms.values())):
            raise RuntimeError("Non-finite internal attention LoRA weights")
        if not any(b_norms.values()):
            raise RuntimeError("No internal attention adapter was updated")
        logger.info("Trained internal LoRA B norms: %s", b_norms)
        self._verify_inputs(self.config, allow_create=False)
        export_merged_checkpoint(model, self._source, self._destination)
        logger.info(
            "Merged pretrained checkpoint with trained adapters: %s", self._destination
        )
