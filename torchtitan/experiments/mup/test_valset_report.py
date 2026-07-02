# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import torch

from . import valset_report as vr


def _fake_batches():
    torch.manual_seed(0)
    for _ in range(3):
        x = torch.randn(4, 5)
        y = torch.randn(4, 5)
        yield {"img": x}, {"plan": y}


def _fake_model(inputs):
    return {"plan": inputs["img"]}


def _fake_loss(pred, targets):
    per_sample = (pred["plan"] - targets["plan"]).pow(2).flatten(1).mean(1)
    return per_sample, {"plan": per_sample, "loss": per_sample}


def _reference(batches):
    losses = []
    for inputs, targets in batches:
        losses.extend(
            (_fake_model(inputs)["plan"] - targets["plan"]).pow(2).flatten(1).mean(1).tolist()
        )
    return sum(losses) / len(losses), len(losses)


def test_atomic_lists_present():
    assert set(vr.ATOMIC_VALSETS) == {
        "day_straight",
        "night_straight",
        "left_lane_change",
        "right_lane_change",
    }
    for name in vr.ATOMIC_VALSETS:
        path = vr.valset_path(name)
        assert os.path.exists(path), path
        assert sum(1 for line in open(path) if line.strip()) > 0


def test_dataloader_config_points_at_valset():
    for name in vr.ATOMIC_VALSETS:
        cfg = vr.valset_dataloader_config(name)
        assert cfg.dataset == vr.valset_path(name)
        assert cfg.split == "val"


def test_accumulate_matches_reference():
    got = vr.accumulate(_fake_model, _fake_loss, _fake_batches(), "cpu")
    ref_loss, ref_n = _reference(_fake_batches())
    assert got["n_samples"] == ref_n
    assert abs(got["loss"] - ref_loss) < 1e-6
    assert abs(got["plan"] - ref_loss) < 1e-6


def test_accumulate_respects_max_steps():
    got = vr.accumulate(_fake_model, _fake_loss, _fake_batches(), "cpu", max_steps=1)
    assert got["n_samples"] == 4


def test_metric_table_and_report():
    results = {
        "vit_mup_w2048": {
            "day_straight": {"loss": 1.0, "plan": 1.0, "n_samples": 512},
            "night_straight": {"loss": 2.0, "plan": 2.0, "n_samples": 512},
        }
    }
    cols, rows = vr.metric_table(results)
    assert cols[:2] == ["run", "valset"]
    assert {r[1] for r in rows} == {"day_straight", "night_straight"}
    with tempfile.TemporaryDirectory() as d:
        out = vr.build_report(results, report_dir=d, report_name="valset_metrics")
        assert os.path.exists(out)
        assert "day_straight" in open(out).read()
