# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate

from torchtitan.observability.sdc_replayer import (
    _compare_signature,
    _find_signature_mismatch,
    _hash_tensor,
    _ReplaySignature,
    _ReplayStateProvider,
    ScalarStateAccessor,
    SDCReplayer,
    SDCReplayMismatch,
)


class _BufferModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("counter", torch.tensor([0.0]), persistent=False)
        self.register_buffer("optional", None, persistent=False)


class _Counter:
    __slots__ = ("value",)

    def __init__(self, value: int = 0):
        self.value = value


def _scalar_state(counter: _Counter) -> dict[str, ScalarStateAccessor]:
    return {
        "ntokens_seen": ScalarStateAccessor(
            get=lambda: counter.value,
            set=lambda value: setattr(counter, "value", value),
        )
    }


def test_state_provider_restores_rng_buffers_gradients_and_counter():
    module = _BufferModule()
    module.weight.grad = torch.zeros(1)
    counter = _Counter(4)
    provider = _ReplayStateProvider(
        [module], torch.device("cpu"), _scalar_state(counter)
    )

    random.seed(7)
    torch.manual_seed(11)
    state = provider.capture()
    expected_python = random.random()
    expected_torch = torch.rand(1)

    module.counter.add_(5)
    module.optional = torch.ones(1)
    module.weight.grad.add_(6)
    counter.value = 99
    random.random()
    torch.rand(1)

    provider.restore(state)

    torch.testing.assert_close(module.counter, torch.zeros(1))
    assert module.optional is None
    torch.testing.assert_close(module.weight.grad, torch.zeros(1))
    assert counter.value == 4
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(1), expected_torch)


def test_replay_commits_only_final_candidate():
    module = _BufferModule()
    counter = _Counter(0)
    replayer = SDCReplayer(
        SDCReplayer.Config(num_replays=2),
        modules=[module],
        device=torch.device("cpu"),
        scalar_state=_scalar_state(counter),
    )
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        module.counter.add_(1)
        counter.value += 5
        loss = module.weight.square().sum()
        loss.backward()
        return loss

    loss = replayer.run_fwd_bwd(execute, step=3)

    assert calls == 3
    torch.testing.assert_close(loss, torch.tensor(4.0))
    torch.testing.assert_close(module.counter, torch.ones(1))
    torch.testing.assert_close(module.weight.grad, torch.tensor([4.0]))
    assert counter.value == 5


def test_replay_wraps_compiled_forward_backward():
    module = torch.nn.Linear(4, 2)
    compiled = torch.compile(module, backend="eager")
    replayer = SDCReplayer(
        SDCReplayer.Config(),
        modules=[module],
        device=torch.device("cpu"),
    )

    def execute():
        loss = compiled(torch.ones(3, 4)).sum()
        loss.backward()
        return loss

    replayer.run_fwd_bwd(execute, step=1)

    assert module.weight.grad is not None


def test_replay_ignores_unregistered_scratch_state():
    module = _BufferModule()
    replayer = SDCReplayer(
        SDCReplayer.Config(),
        modules=[module],
        device=torch.device("cpu"),
    )
    scratch_invocations = 0

    def execute():
        nonlocal scratch_invocations
        scratch_invocations += 1
        loss = module.weight.square().sum()
        loss.backward()
        return loss

    replayer.run_fwd_bwd(execute, step=1)

    assert scratch_invocations == 2


def test_replay_mismatch_is_fatal():
    module = _BufferModule()
    replayer = SDCReplayer(
        SDCReplayer.Config(),
        modules=[module],
        device=torch.device("cpu"),
    )
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        loss = module.weight.square().sum()
        loss.backward()
        if calls == 2:
            module.weight.grad.add_(1)
        return loss

    with pytest.raises(SDCReplayMismatch, match="gradient:0:weight"):
        replayer.run_fwd_bwd(execute, step=9)
    assert replayer.steps_since_reset == 0


@pytest.mark.parametrize(
    ("corruption", "expected"),
    (
        ("buffer", "buffer:0:counter"),
        ("cpu_rng", "rng:cpu"),
        ("python_rng", "state:python_rng"),
        ("counter", "state:ntokens_seen"),
    ),
)
def test_replay_detects_semantic_state_mismatch(corruption, expected):
    module = _BufferModule()
    counter = _Counter(0)
    replayer = SDCReplayer(
        SDCReplayer.Config(),
        modules=[module],
        device=torch.device("cpu"),
        scalar_state=_scalar_state(counter),
    )
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        loss = module.weight.square().sum()
        loss.backward()
        if calls == 2:
            if corruption == "buffer":
                module.counter.add_(1)
            elif corruption == "cpu_rng":
                torch.rand(1)
            elif corruption == "python_rng":
                random.random()
            else:
                counter.value += 1
        return loss

    with pytest.raises(SDCReplayMismatch, match=expected):
        replayer.run_fwd_bwd(execute, step=9)


@pytest.mark.parametrize(
    ("num_steps", "expected_checked"),
    [
        (1, [True, False, False, False]),
        (3, [True, True, True, False]),
        (-1, [True, True, True, True]),
    ],
)
def test_schedule_checks_first_num_steps_and_rearms_on_reset(
    num_steps, expected_checked
):
    replayer = SDCReplayer(
        SDCReplayer.Config(num_steps=num_steps),
        modules=[],
        device=torch.device("cpu"),
    )
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        return torch.tensor(1.0)

    checked = []
    for index in range(len(expected_checked)):
        calls_before = calls
        replayer.run_fwd_bwd(execute, step=index)
        # A checked step executes twice (reference plus one replay).
        checked.append(calls - calls_before == 2)
        assert replayer.steps_since_reset == index + 1
    assert checked == expected_checked

    replayer.reset_schedule()
    assert replayer.steps_since_reset == 0
    calls_before = calls
    replayer.run_fwd_bwd(execute, step=99)
    assert calls - calls_before == 2


def test_config_validation():
    with pytest.raises(ValueError, match="num_steps"):
        SDCReplayer.Config(num_steps=0)
    with pytest.raises(ValueError, match="num_replays"):
        SDCReplayer.Config(num_replays=0)


def test_empty_tensor_hash_uses_zero_sentinel():
    digest = _hash_tensor(torch.empty(0))

    assert digest.dtype == torch.uint64
    assert digest.item() == 0


def test_complex_tensor_hash_uses_real_view():
    value = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)

    digest = _hash_tensor(value)
    expected = torch.hash_tensor(torch.view_as_real(value))

    torch.testing.assert_close(digest, expected)
    assert digest.dtype == torch.uint64


def test_signature_digest_comparison_is_batched():
    schema = (
        ("loss", (), "torch.float32", "cpu"),
        ("gradient:0:weight", (), "torch.float32", "cpu"),
    )
    reference = _ReplaySignature(
        schema=schema,
        digests=(
            torch.tensor(1, dtype=torch.uint64),
            torch.tensor(2, dtype=torch.uint64),
        ),
        state=(),
    )
    candidate = _ReplaySignature(
        schema=schema,
        digests=(
            torch.tensor(1, dtype=torch.uint64),
            torch.tensor(3, dtype=torch.uint64),
        ),
        state=(),
    )

    with patch("torchtitan.observability.sdc_replayer.torch.equal") as equal:
        mismatch = _compare_signature(reference, candidate)

    assert bool(mismatch.item())
    equal.assert_not_called()
    assert _find_signature_mismatch(reference, candidate) == "gradient:0:weight"


def test_state_provider_restores_dtensor_local_shards(tmp_path):
    assert not dist.is_initialized()
    store_path = tmp_path / "store"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        module = _BufferModule()
        module.weight = torch.nn.Parameter(
            distribute_tensor(module.weight.detach(), mesh, [Replicate()])
        )
        module.counter = distribute_tensor(module.counter, mesh, [Replicate()])
        module.weight.grad = distribute_tensor(torch.zeros(1), mesh, [Replicate()])
        provider = _ReplayStateProvider([module], torch.device("cpu"), {})
        state = provider.capture()

        module.counter.to_local().add_(5)
        module.weight.grad.to_local().add_(6)
        provider.restore(state)

        torch.testing.assert_close(module.counter.to_local(), torch.zeros(1))
        torch.testing.assert_close(module.weight.grad.to_local(), torch.zeros(1))
    finally:
        dist.destroy_process_group()


def test_remote_mismatch_is_fatal_on_a_locally_matching_rank():
    replayer = SDCReplayer(
        SDCReplayer.Config(),
        modules=[],
        device=torch.device("cpu"),
    )

    def report_global_mismatch(tensor, **kwargs):
        tensor.fill_(1)

    def gather_remote_details(output, local_details):
        output[:] = [(0, "loss"), local_details]

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_reduce", side_effect=report_global_mismatch),
        patch(
            "torch.distributed.all_gather_object",
            side_effect=gather_remote_details,
        ),
        pytest.raises(SDCReplayMismatch, match="rank=0"),
    ):
        signature = _ReplaySignature(schema=(), digests=(), state=())
        replayer._raise_if_mismatch(
            step=2,
            local_step=1,
            replay=1,
            local_mismatch=torch.tensor(False),
            reference=signature,
            candidate=signature,
        )
