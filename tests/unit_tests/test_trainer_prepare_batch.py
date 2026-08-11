# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.trainer import Trainer


def test_prepare_batch_accumulates_local_ntokens_and_returns_triple():
    captured = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, labels, **kw):
            captured.update(kw)
            return ("INPUTS", "LABELS", {"positions": 1}, 7)

    fake = SimpleNamespace(
        model_parts=[_FakeModel()],
        parallel_dims="PD",
        device="cpu",
        config=SimpleNamespace(parallelism="PARA"),
        ntokens_seen=100,
    )
    out = Trainer._prepare_batch(fake, {"input": 0}, "labels")
    assert out == ("INPUTS", "LABELS", {"positions": 1})  # 3-tuple, 4th dropped
    assert fake.ntokens_seen == 107  # local_ntokens (7) folded in
    assert captured == {"parallel_dims": "PD", "device": "cpu", "parallelism": "PARA"}
