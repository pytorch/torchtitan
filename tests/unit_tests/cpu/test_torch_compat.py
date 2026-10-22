# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral regressions for the release's upstream pipelining backports."""

import copy
import os
import subprocess
import sys

import pytest
import torch
import torchtitan  # noqa: F401
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
)
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from torch.testing._internal.distributed.fake_pg import FakeStore


@pytest.fixture
def process_group():
    torch.manual_seed(42)
    torch.distributed.init_process_group(
        backend="fake", rank=0, world_size=1, store=FakeStore()
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def test_metadata_cleanup_lifecycle(process_group):
    device = torch.device("cpu")
    x = torch.randn(2, 4)
    target = torch.randn_like(x)
    stage = PipelineStage(torch.nn.Linear(4, 4), 0, 1, device)
    schedule = ScheduleGPipe(stage, 1, loss_fn=torch.nn.MSELoss())
    calls = []

    def cleanup():
        assert stage._metadata_inference_buffer_backup is None
        assert stage._fwd_outputs_for_bwd_meta is None
        calls.append("cleanup")

    handle = schedule.register_post_metadata_inference_cleanup(cleanup)
    schedule.step(x, target=target)
    schedule.step(x, target=target)
    assert calls == ["cleanup"]
    handle.remove()
    schedule.eval(x, target=target)
    assert calls == ["cleanup"]

    module = torch.nn.Linear(4, 4)
    static_stage = PipelineStage(
        module, 0, 1, device, input_args=x, output_args=module(x)
    )
    static_schedule = ScheduleGPipe(static_stage, 1)
    static_schedule.register_post_metadata_inference_cleanup(
        lambda: calls.append("static")
    )
    static_schedule.step(x)
    assert calls == ["cleanup"]


@pytest.mark.parametrize(
    "schedule_class",
    [
        ScheduleGPipe,
        Schedule1F1B,
        ScheduleInterleaved1F1B,
        ScheduleInterleavedZeroBubble,
        ScheduleLoopedBFS,
    ],
)
def test_microbatch_loss_kwargs_and_gradients(process_group, schedule_class):
    module = torch.nn.Linear(16, 16)
    reference = copy.deepcopy(module)
    x_mbs = torch.randn(8, 16).chunk(2)
    target_mbs = list(torch.randn(8, 16).chunk(2))
    weights = torch.rand(8, 1).chunk(2)

    def loss_fn(output, target, *, scale, weight, metrics):
        loss = ((output - target).square() * weight).sum() * scale
        metrics["loss"] = loss.detach()
        return loss

    stage = PipelineStage(module, 0, 1, torch.device("cpu"))
    stages = stage if issubclass(schedule_class, PipelineScheduleSingle) else [stage]
    schedule = schedule_class(stages, 2, loss_fn=loss_fn, scale_grads=False)
    metrics = [{}, {}]
    losses = []
    schedule.step(
        arg_mbs=[(x,) for x in x_mbs],
        target_mbs=target_mbs,
        loss_kwargs={"scale": 0.5},
        loss_kwarg_mbs=[
            {"weight": w, "metrics": m} for w, m in zip(weights, metrics, strict=True)
        ],
        losses=losses,
    )
    for i, (x, target, weight) in enumerate(
        zip(x_mbs, target_mbs, weights, strict=True)
    ):
        loss = loss_fn(reference(x), target, scale=0.5, weight=weight, metrics={})
        loss.backward()
        torch.testing.assert_close(losses[i], loss.detach(), rtol=0, atol=0)
        torch.testing.assert_close(metrics[i]["loss"], loss.detach(), rtol=0, atol=0)
    for name, param in module.named_parameters():
        torch.testing.assert_close(param.grad, reference.get_parameter(name).grad)
    assert schedule._loss_kwarg_mbs is None

    with pytest.raises(TypeError, match="multiple values for keyword argument 'scale'"):
        schedule.step(
            arg_mbs=[(x,) for x in x_mbs],
            target_mbs=target_mbs,
            loss_kwargs={"scale": 0.5},
            loss_kwarg_mbs=[{"scale": 1.0}, {"scale": 2.0}],
        )
    assert schedule._loss_kwarg_mbs is None


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"loss_kwarg_mbs": {}}, TypeError, "must be a list"),
        ({"loss_kwarg_mbs": [{}]}, ValueError, "Expecting 2 loss_kwarg_mbs"),
        ({"loss_kwarg_mbs": [1, 2]}, TypeError, "must be a list of dicts"),
    ],
)
def test_microbatch_loss_kwargs_validation(process_group, kwargs, error, match):
    stage = PipelineStage(torch.nn.Linear(4, 4), 0, 1, torch.device("cpu"))
    schedule = ScheduleGPipe(stage, 2, loss_fn=torch.nn.MSELoss())
    with pytest.raises(error, match=match):
        schedule.step(arg_mbs=[(torch.randn(2, 4),)] * 2, **kwargs)


def test_split_backward_view_output():
    class Identity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return grad

    torch.manual_seed(42)
    module = torch.nn.Sequential(
        torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16)
    )
    reference = copy.deepcopy(module)
    x = torch.randn(8, 16, requires_grad=True)
    ref_x = x.detach().requires_grad_(True)
    grad = torch.randn_like(x)
    output = Identity.apply(module(x))
    assert output._is_view()
    dinputs, groups = stage_backward_input((output,), [grad], [x], module.parameters())
    stage_backward_weight(module.parameters(), groups)
    reference(ref_x).backward(grad)
    torch.testing.assert_close(dinputs[0], ref_x.grad, rtol=0, atol=0)
    for name, param in module.named_parameters():
        torch.testing.assert_close(
            param.grad, reference.get_parameter(name).grad, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "mode", ["disabled", "other_version", "bad_source", "already_patched"]
)
def test_import_gates_and_atomicity(mode):
    script = r"""
import os
import sys
from unittest.mock import patch

os.environ["TORCHTITAN_DISABLE_TORCH_COMPAT"] = "1"
from torch.distributed.pipelining import schedules
import torchtitan
from torchtitan._torch_compat import apply_patches
from torchtitan._torch_compat._pipelining_edits import PATCHES

original = schedules._PipelineSchedule.__init__
assert apply_patches() == ()
mode = sys.argv[1]
if mode == "other_version":
    del os.environ["TORCHTITAN_DISABLE_TORCH_COMPAT"]
    with patch("torchtitan._torch_compat.version", return_value="2.16.0.dev20261020"):
        assert apply_patches() == ()
elif mode == "bad_source":
    del os.environ["TORCHTITAN_DISABLE_TORCH_COMPAT"]
    PATCHES[1]["functions"][0]["before_sha256"] = "unsupported"
    # Require #200238 even when this torch wheel already contains the fix.
    with patch.object(schedules.PipelineScheduleSingle, "step", lambda self: None):
        try:
            apply_patches()
        except RuntimeError as exc:
            assert "Unsupported torch" in str(exc)
        else:
            raise AssertionError("Unknown source was accepted")
elif mode == "already_patched":
    del os.environ["TORCHTITAN_DISABLE_TORCH_COMPAT"]
    apply_patches()
    original = schedules._PipelineSchedule.__init__
    original_step = schedules.PipelineScheduleSingle.step
    # Recheck the installed features instead of returning the cached result.
    with patch("torchtitan._torch_compat._applied_patches", None):
        assert apply_patches() == ()
    assert schedules.PipelineScheduleSingle.step is original_step
assert schedules._PipelineSchedule.__init__ is original
"""
    if (
        mode in ("bad_source", "already_patched")
        and torch.__version__.split("+")[0] != "2.15.0"
    ):
        pytest.skip("Backport application checks target the validated 2.15 RC")
    subprocess.run(
        [sys.executable, "-c", script, mode],
        check=True,
        env=os.environ.copy(),
        timeout=60,
    )
