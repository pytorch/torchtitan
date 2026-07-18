# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable dataset recipes backed by Grain."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeAlias

import grain.python as grain
import numpy as np

from torchtitan.components.data.sources import RandomAccessSource, SourceConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset
ChatMessages: TypeAlias = list[dict[str, Any]]
SampleToMessages: TypeAlias = Callable[[dict[str, Any]], ChatMessages]


@dataclass(frozen=True, kw_only=True, slots=True)
class DataRuntime:
    """Objects and sizes available while a dataset recipe is built."""

    tokenizer: BaseTokenizer
    seq_len: int
    local_batch_size: int
    read_options: grain.ReadOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class BuildOptions:
    """Run policy supplied once by the loader."""

    seed: int = 0
    shuffle: bool = True
    repeat: bool = True
    dp_rank: int = 0
    dp_world_size: int = 1


class DatasetConfig(Protocol):
    """Builds one node of a Grain dataset graph."""

    def build(self, *, runtime: DataRuntime, options: BuildOptions) -> GrainDataset:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenSequence:
    """One tokenized document and the positions that train the loss."""

    token_ids: np.ndarray
    loss_mask: np.ndarray


class SampleProcessor(Configurable, ABC):
    """Configured row processor using Grain-provided deterministic randomness."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(
        self,
        sample: Any,
        rng: np.random.Generator,
    ) -> Any:
        ...


class TextToTokenSequence(SampleProcessor):
    """Tokenizes one text field for causal-language-model pretraining."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        text_field: str = "text"

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        self._text_field = config.text_field
        self._tokenizer = runtime.tokenizer

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> TokenSequence:
        del rng
        token_ids = np.asarray(
            self._tokenizer.encode(
                sample[self._text_field],
                add_bos=True,
                add_eos=True,
            ),
            dtype=np.int64,
        )
        return TokenSequence(
            token_ids=token_ids,
            loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
        )


class ChatToTokenSequence(SampleProcessor):
    """Applies a chat template and marks the assistant response for training."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        sample_to_messages: SampleToMessages
        train_on_assistant_only: bool = True

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        if runtime.tokenizer.eos_id is None:
            raise ValueError("ChatToTokenSequence requires a tokenizer EOS token")
        self._sample_to_messages = config.sample_to_messages
        self._train_on_assistant_only = config.train_on_assistant_only
        self._tokenizer = runtime.tokenizer
        self._eos_id = runtime.tokenizer.eos_id

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> TokenSequence:
        del rng
        messages = self._sample_to_messages(sample)
        _validate_single_turn_messages(messages)

        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        token_ids = self._tokenizer.encode(
            full_text,
            add_bos=True,
            add_eos=False,
        )
        if token_ids[-1] != self._eos_id:
            token_ids.append(self._eos_id)

        loss_mask = np.ones(len(token_ids), dtype=np.bool_)
        if self._train_on_assistant_only:
            prompt_text = self._tokenizer.apply_chat_template(
                messages[:1],
                add_generation_prompt=True,
            )
            prompt_length = len(
                self._tokenizer.encode(
                    prompt_text,
                    add_bos=True,
                    add_eos=False,
                )
            )
            loss_mask[:prompt_length] = False

        return TokenSequence(
            token_ids=np.asarray(token_ids, dtype=np.int64),
            loss_mask=loss_mask,
        )


def _validate_single_turn_messages(messages: ChatMessages) -> None:
    if len(messages) != 2:
        raise ValueError(
            f"expected one user and one assistant message, got {len(messages)}"
        )
    if messages[0]["role"] != "user":
        raise ValueError(
            f"first message must have role 'user', got {messages[0]['role']!r}"
        )
    if messages[1]["role"] != "assistant":
        raise ValueError(
            f"second message must have role 'assistant', got {messages[1]['role']!r}"
        )


def _apply_process(
    dataset: GrainDataset,
    process: SampleProcessor.Config | Callable[[Any], Any] | None,
    *,
    runtime: DataRuntime,
    seed: int,
) -> GrainDataset:
    if process is None:
        return dataset
    if isinstance(process, SampleProcessor.Config):
        processor = process.build(runtime=runtime)
        return dataset.random_map(processor, seed=seed)
    return dataset.map(process)


def _finish_random_access(
    dataset: grain.MapDataset,
    *,
    options: BuildOptions,
) -> grain.MapDataset:
    if options.shuffle:
        dataset = dataset.shuffle(seed=options.seed)
    if len(dataset) < options.dp_world_size:
        raise ValueError(
            f"dataset has {len(dataset)} rows, fewer than "
            f"dp_world_size={options.dp_world_size}"
        )
    dataset = dataset[options.dp_rank :: options.dp_world_size]
    if options.repeat:
        dataset = dataset.repeat()
    return dataset


def _finish_streaming(
    dataset: grain.IterDataset,
    *,
    options: BuildOptions,
    shuffle_window_size: int,
) -> grain.IterDataset:
    if options.repeat:
        dataset = grain.experimental.RepeatIterDataset(dataset)
    if options.shuffle:
        dataset = grain.experimental.WindowShuffleIterDataset(
            dataset,
            window_size=shuffle_window_size,
            seed=options.seed,
        )
    return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDatasetConfig:
    """One source with row processing, filtering, and run policy."""

    source: SourceConfig
    process: SampleProcessor.Config | Callable[[Any], Any] | None = None
    filters: tuple[Callable[[Any], bool], ...] = ()
    shuffle_window_size: int = 10_000

    def build(self, *, runtime: DataRuntime, options: BuildOptions) -> GrainDataset:
        if self.filters and not options.repeat and options.dp_world_size > 1:
            raise ValueError(
                "finite filtered datasets are not supported with data parallelism"
            )
        source = self.source.build(runtime=runtime, options=options)
        if isinstance(source, grain.IterDataset):
            dataset = _apply_process(
                source,
                self.process,
                runtime=runtime,
                seed=options.seed + options.dp_rank,
            )
            for filter_fn in self.filters:
                dataset = dataset.filter(filter_fn)
            return _finish_streaming(
                dataset,
                options=options,
                shuffle_window_size=self.shuffle_window_size,
            )

        if not isinstance(source, RandomAccessSource):
            raise TypeError("source must support len/getitem or be a Grain IterDataset")
        dataset = grain.MapDataset.source(source)
        dataset = _apply_process(
            dataset,
            self.process,
            runtime=runtime,
            seed=options.seed,
        )
        for filter_fn in self.filters:
            dataset = dataset.filter(filter_fn)
        return _finish_random_access(dataset, options=options)


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: DatasetConfig
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMixConfig:
    """Deterministically interleaves weighted child datasets."""

    datasets: tuple[WeightedDataset, ...]

    def build(self, *, runtime: DataRuntime, options: BuildOptions) -> GrainDataset:
        if not self.datasets or any(item.weight <= 0 for item in self.datasets):
            raise ValueError("DatasetMixConfig requires positive-weight datasets")
        children = [
            item.dataset.build(
                runtime=runtime,
                options=replace(options, seed=options.seed + index),
            )
            for index, item in enumerate(self.datasets)
        ]
        weights = [item.weight for item in self.datasets]
        if all(isinstance(child, grain.MapDataset) for child in children):
            return grain.MapDataset.mix(children, weights=weights)
        if all(isinstance(child, grain.IterDataset) for child in children):
            return grain.IterDataset.mix(children, weights=weights)
        raise TypeError("a mix cannot combine map and iterable child datasets")


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcatConfig:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[DatasetConfig, ...]

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.MapDataset:
        finite_options = replace(
            options,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
        )
        children = [
            dataset.build(runtime=runtime, options=finite_options)
            for dataset in self.datasets
        ]
        if not children or not all(
            isinstance(child, grain.MapDataset) for child in children
        ):
            raise TypeError("DatasetConcatConfig requires map-style children")
        combined = grain.MapDataset.concatenate(children)
        return _finish_random_access(combined, options=options)
