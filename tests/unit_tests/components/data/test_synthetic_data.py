# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from torchtitan.components.data import BucketLengthSpec, LengthBucket


def test_bucket_spec_lengths_stay_within_bucket_ranges():
    spec = BucketLengthSpec(
        buckets=(
            LengthBucket(min_len=1, max_len=4, weight=1.0),
            LengthBucket(min_len=100, max_len=128, weight=3.0),
        )
    )
    lengths = spec.sample(np.random.default_rng(0), 5000)

    assert lengths.dtype == np.int64
    in_low = (lengths >= 1) & (lengths <= 4)
    in_high = (lengths >= 100) & (lengths <= 128)
    assert np.all(in_low | in_high)
    # weight 3:1 -> ~75% land in the high bucket (allow slack).
    assert 0.68 < in_high.mean() < 0.82


def test_bucket_spec_is_reproducible_for_same_seed():
    spec = BucketLengthSpec(buckets=(LengthBucket(min_len=1, max_len=50),))
    a = spec.sample(np.random.default_rng(7), 100)
    b = spec.sample(np.random.default_rng(7), 100)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_len": 0, "max_len": 4},
        {"min_len": 5, "max_len": 4},
        {"min_len": 1, "max_len": 4, "weight": 0.0},
    ],
)
def test_length_bucket_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        LengthBucket(**kwargs)


def test_bucket_spec_requires_at_least_one_bucket():
    with pytest.raises(ValueError):
        BucketLengthSpec(buckets=())


from torchtitan.components.data import ParametricLengthSpec


def test_parametric_uniform_within_bounds():
    spec = ParametricLengthSpec(kind="uniform", min_len=8, max_len=64)
    lengths = spec.sample(np.random.default_rng(0), 5000)
    assert lengths.dtype == np.int64
    assert lengths.min() >= 8
    assert lengths.max() <= 64


def test_parametric_normal_clamps_to_bounds():
    spec = ParametricLengthSpec(
        kind="normal", min_len=1, max_len=100, mean=50.0, std=1000.0
    )
    lengths = spec.sample(np.random.default_rng(0), 5000)
    assert lengths.min() >= 1
    assert lengths.max() <= 100
    # Huge std under clamping -> mass piles at both bounds.
    assert (lengths == 1).any() and (lengths == 100).any()


def test_parametric_lognormal_is_right_skewed():
    spec = ParametricLengthSpec(
        kind="lognormal", min_len=1, max_len=8192, mean=5.0, std=1.0
    )
    lengths = spec.sample(np.random.default_rng(0), 20000)
    assert np.median(lengths) < lengths.mean()  # right tail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "normal", "min_len": 1, "max_len": 10},  # missing mean/std
        {"kind": "zipf", "min_len": 1, "max_len": 10, "alpha": 1.0},  # alpha<=1
        {"kind": "uniform", "min_len": 5, "max_len": 4},  # bad bounds
    ],
)
def test_parametric_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        ParametricLengthSpec(**kwargs)


from torchtitan.components.data import SyntheticLengthSource
from torchtitan.components.data.types import DatasetIterationPolicy


def _policy(**overrides):
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": True,
        "dp_rank": 0,
        "dp_world_size": 1,
        "streaming_shuffle_buffer_size": 4,
    }
    return DatasetIterationPolicy(**(values | overrides))


def _source(**policy_overrides):
    return SyntheticLengthSource.Config(
        length_spec=BucketLengthSpec(buckets=(LengthBucket(min_len=1, max_len=64),)),
        seed=0,
    ).build(dataset_iteration_policy=_policy(**policy_overrides))


def test_synthetic_source_emits_lengths_in_range():
    it = iter(_source())
    lengths = [next(it)["length"] for _ in range(1000)]
    assert all(1 <= n <= 64 for n in lengths)


def test_synthetic_source_same_policy_is_reproducible():
    it_a, it_b = iter(_source()), iter(_source())
    seq_a = [next(it_a)["length"] for _ in range(50)]
    seq_b = [next(it_b)["length"] for _ in range(50)]
    assert seq_a == seq_b


def test_synthetic_source_differs_by_dp_rank():
    it0 = iter(_source(dp_rank=0, dp_world_size=2))
    it1 = iter(_source(dp_rank=1, dp_world_size=2))
    seq0 = [next(it0)["length"] for _ in range(50)]
    seq1 = [next(it1)["length"] for _ in range(50)]
    assert seq0 != seq1


def test_synthetic_source_resumes_exactly():
    it = iter(_source())
    for _ in range(10):
        next(it)
    state = it.get_state()
    expected = [next(it)["length"] for _ in range(20)]

    restored = iter(_source())
    restored.set_state(state)
    assert [next(restored)["length"] for _ in range(20)] == expected


def test_synthetic_source_resumes_across_chunk_boundary():
    it = iter(_source())
    for _ in range(1030):  # past the default 1024 chunk
        next(it)
    state = it.get_state()
    expected = [next(it)["length"] for _ in range(2100)]

    restored = iter(_source())
    restored.set_state(state)
    assert [next(restored)["length"] for _ in range(2100)] == expected


from dataclasses import replace

import grain.python as grain

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data import RandomTokenProcessor
from torchtitan.components.data.types import DatasetBuildContext


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [ord(char) % 250 + 10 for char in text]
        return [self.bos_id] * add_bos + tokens + [self.eos_id] * add_eos


CONTEXT = DatasetBuildContext(
    tokenizer=FakeTokenizer(),
    max_context_length=9,
    num_tokens_per_batch=18,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def _processor(vocab_size=100, max_context_length=128):
    ctx = replace(CONTEXT, max_context_length=max_context_length)
    return RandomTokenProcessor.Config(vocab_size=vocab_size).build(context=ctx)


def test_random_token_processor_builds_aligned_pairs():
    proc = _processor()
    seq = proc({"length": 10}, np.random.default_rng(0))
    assert isinstance(seq, TextSequence)
    assert seq.input_ids.shape == (10,)
    assert seq.labels.shape == (10,)
    assert seq.input_ids.dtype == np.int64
    assert (seq.input_ids >= 0).all() and (seq.input_ids < 100).all()


def test_random_token_processor_drops_oversized():
    proc = _processor(max_context_length=16)
    assert proc({"length": 16}, np.random.default_rng(0)) is None  # 16+1 > 16
    assert proc({"length": 15}, np.random.default_rng(0)) is not None


def test_random_token_processor_uses_tokenizer_vocab_when_unset():
    class VocabTokenizer:
        def get_vocab_size(self):
            return 7

    ctx = replace(CONTEXT, tokenizer=VocabTokenizer(), max_context_length=128)
    proc = RandomTokenProcessor.Config().build(context=ctx)
    seq = proc({"length": 20}, np.random.default_rng(0))
    assert (seq.input_ids < 7).all()


from torchtitan.components.data import ConstantTokenProcessor


def _const_processor(constant_token_id=5, max_context_length=128):
    ctx = replace(CONTEXT, max_context_length=max_context_length)
    return ConstantTokenProcessor.Config(constant_token_id=constant_token_id).build(
        context=ctx
    )


def test_constant_processor_fills_constant_pairs():
    proc = _const_processor(constant_token_id=5)
    seq = proc({"length": 10}, np.random.default_rng(0))
    assert isinstance(seq, TextSequence)
    assert seq.input_ids.shape == (10,)
    assert seq.labels.shape == (10,)
    assert seq.input_ids.dtype == np.int64
    assert (seq.input_ids == 5).all() and (seq.labels == 5).all()


def test_constant_processor_drops_oversized():
    proc = _const_processor(max_context_length=16)
    assert proc({"length": 16}, np.random.default_rng(0)) is None
    assert proc({"length": 15}, np.random.default_rng(0)) is not None


def test_constant_processor_rejects_negative_id():
    ctx = replace(CONTEXT, max_context_length=128)
    with pytest.raises(ValueError):
        ConstantTokenProcessor.Config(constant_token_id=-1).build(context=ctx)


import torch

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader


def _loader():
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=SyntheticLengthSource.Config(
                length_spec=BucketLengthSpec(
                    buckets=(LengthBucket(min_len=2, max_len=6),)
                ),
                seed=0,
            ),
            processor=RandomTokenProcessor.Config(vocab_size=100),
            post_filters=(lambda s: s is not None,),
        ),
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
        num_prefetch_batches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        max_context_length=8,
        num_tokens_per_batch=16,
    )


def test_synthetic_pipeline_produces_batches():
    inputs, labels = next(iter(_loader()))
    assert "input" in inputs and "positions" in inputs
    assert inputs["input"].shape[-1] == 16  # num_tokens_per_batch
    assert labels.shape[-1] == 16


def test_synthetic_pipeline_resumes_exactly():
    loader = _loader()
    it = iter(loader)
    next(it)
    state = loader.state_dict()
    expected = next(it)

    restored = _loader()
    restored.load_state_dict(state)
    actual = next(iter(restored))

    assert torch.equal(expected[0]["input"], actual[0]["input"])
    assert torch.equal(expected[1], actual[1])
