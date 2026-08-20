# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.components.validate import Validator


def test_prepare_batch_forwards_and_drops_ntokens():
    captured = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, labels, **kw):
            captured.update(kw)
            return ("INPUTS", "LABELS", {"positions": 1}, 7)  # 4-tuple

    fake = SimpleNamespace(parallel_dims="PD", parallelism="PARA")
    input_dict = {"input": SimpleNamespace(device="cpu")}
    out = Validator._prepare_batch(fake, input_dict, "labels", [_FakeModel()])
    assert out == ("INPUTS", "LABELS", {"positions": 1})  # 3-tuple, 4th dropped
    assert captured == {"parallel_dims": "PD", "device": "cpu", "parallelism": "PARA"}
