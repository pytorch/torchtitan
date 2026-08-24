# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scripts.preview_synthetic_lengths import build_spec, dp_balance, summarize


def test_build_spec_buckets():
    spec = build_spec(
        {"type": "buckets", "buckets": [{"min_len": 1, "max_len": 4, "weight": 2.0}]}
    )
    lengths = spec.sample(np.random.default_rng(0), 100)
    assert lengths.min() >= 1 and lengths.max() <= 4


def test_summarize_reports_percentiles():
    stats = summarize(np.arange(1, 101))
    assert stats["p50"] == 50.5  # np.percentile linear interpolation over 100 elems
    assert stats["max"] == 100
    assert stats["mean"] == 50.5
    assert stats["count"] == 100


def test_dp_balance_flags_imbalance():
    # Rank 0 always long, rank 1 always short -> imbalance ratio > 1.
    lengths = np.array([100, 1] * 50)
    report = dp_balance(lengths, dp_world_size=2, per_rank_batch=1)
    assert report["max_over_mean"] > 1.5


def test_summarize_handles_empty():
    stats = summarize(np.array([], dtype=np.int64))
    assert stats["count"] == 0
    assert stats["max"] == 0
    assert stats["total_tokens"] == 0


def test_dp_balance_handles_fewer_samples_than_one_step():
    # 3 samples but a step consumes dp*per_rank = 8 -> no full step.
    report = dp_balance(np.array([5, 6, 7]), dp_world_size=8, per_rank_batch=1)
    assert report["num_steps"] == 0
    assert report["worst_step_max_over_mean"] == 0.0
