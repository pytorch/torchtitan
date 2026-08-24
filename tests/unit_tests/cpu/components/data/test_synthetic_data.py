# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from torchtitan.components.data import BucketLengthSpec, LengthBucket, SyntheticSource
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import DatasetIterationPolicy


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


def _source(
    *, vocab_size=100, min_len=1, max_len=64, source_seed=0, **policy_overrides
):
    return SyntheticSource.Config(
        length_spec=BucketLengthSpec(
            buckets=(LengthBucket(min_len=min_len, max_len=max_len),)
        ),
        vocab_size=vocab_size,
        seed=source_seed,
    ).build(dataset_iteration_policy=_policy(**policy_overrides))


def test_synthetic_source_emits_text_sequences_in_range():
    it = iter(_source())
    for _ in range(1000):
        seq = next(it)
        assert isinstance(seq, TextSequence)
        assert seq.input_ids.dtype == np.int64
        assert 1 <= seq.input_ids.shape[0] <= 64
        assert seq.labels.shape == seq.input_ids.shape
        assert (seq.input_ids >= 0).all() and (seq.input_ids < 100).all()


def test_synthetic_source_same_policy_is_reproducible():
    it_a, it_b = iter(_source()), iter(_source())
    seq_a = [next(it_a).input_ids.tolist() for _ in range(50)]
    seq_b = [next(it_b).input_ids.tolist() for _ in range(50)]
    assert seq_a == seq_b


def test_synthetic_source_differs_by_dp_rank():
    it0 = iter(_source(dp_rank=0, dp_world_size=2))
    it1 = iter(_source(dp_rank=1, dp_world_size=2))
    seq0 = [next(it0).input_ids.tolist() for _ in range(50)]
    seq1 = [next(it1).input_ids.tolist() for _ in range(50)]
    assert seq0 != seq1


def test_synthetic_source_does_not_drop_oversized():
    # The source no longer knows max_context_length; every sampled length is
    # realized (oversize handling is left to FirstFit packing).
    it = iter(_source(min_len=200, max_len=256))
    for _ in range(100):
        assert next(it).input_ids.shape[0] >= 200


def test_synthetic_source_config_rejects_nonpositive_vocab():
    with pytest.raises(ValueError):
        SyntheticSource.Config(
            length_spec=BucketLengthSpec(buckets=(LengthBucket(min_len=1, max_len=8),)),
            vocab_size=0,
        )


def test_synthetic_source_resumes_exactly():
    it = iter(_source())
    for _ in range(10):
        next(it)
    state = it.get_state()
    expected = [next(it).input_ids.tolist() for _ in range(20)]

    restored = iter(_source())
    restored.set_state(state)
    assert [next(restored).input_ids.tolist() for _ in range(20)] == expected


def test_synthetic_source_resumes_across_chunk_boundary():
    it = iter(_source())
    for _ in range(1030):  # past the default 1024 chunk
        next(it)
    state = it.get_state()
    expected = [next(it).input_ids.tolist() for _ in range(2100)]

    restored = iter(_source())
    restored.set_state(state)
    assert [next(restored).input_ids.tolist() for _ in range(2100)] == expected


import torch
from torchtitan.components.data import synthetic_dataloader_builder


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def get_vocab_size(self):
        return 100

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [ord(char) % 250 + 10 for char in text]
        return [self.bos_id] * add_bos + tokens + [self.eos_id] * add_eos


def _loader():
    return synthetic_dataloader_builder(
        length_spec=BucketLengthSpec(buckets=(LengthBucket(min_len=2, max_len=6),)),
        vocab_size=100,
        seed=0,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        max_context_length=8,
        num_tokens_per_batch=16,
    )


def test_synthetic_dataloader_builder_produces_batches():
    inputs, labels = next(iter(_loader()))
    assert "input" in inputs and "positions" in inputs
    assert inputs["input"].shape[-1] == 16  # num_tokens_per_batch
    assert labels.shape[-1] == 16


def test_synthetic_dataloader_builder_resumes_exactly():
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
