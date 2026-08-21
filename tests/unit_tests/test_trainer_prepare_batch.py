# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torchtitan.trainer import Trainer


def test_prepare_batch_accumulates_tokens_and_returns_triple():
    captured = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kw):
            captured.update(kw)
            return ("INPUTS", torch.ones(7), {"positions": 1})

    fake = SimpleNamespace(
        model_parts=[_FakeModel()],
        parallel_dims="PD",
        config=SimpleNamespace(parallelism="PARA"),
        ntokens_seen=100,
    )
    inputs, labels, extra = Trainer._prepare_batch(fake, {"input": 0}, "labels")
    assert inputs == "INPUTS"
    assert extra == {"positions": 1}
    assert labels.numel() == 7
    assert fake.ntokens_seen == 107  # labels.numel() (7) folded in
    assert captured == {"parallel_dims": "PD", "parallelism": "PARA"}
