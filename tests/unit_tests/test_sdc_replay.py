# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate

from torchtitan.distributed.sdc_replay import (
    _compare_signature,
    _find_signature_mismatch,
    _hash_tensor,
    ReplaySignature,
    SDCReplay,
    SDCReplayConfig,
    SDCReplayMismatch,
    TrainerReplayStateProvider,
)


class _BufferModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("counter", torch.tensor([0.0]), persistent=False)
        self.register_buffer("optional", None, persistent=False)


def test_state_provider_restores_rng_buffers_gradients_and_counter():
    module = _BufferModule()
    module.weight.grad = torch.tensor([3.0])
    trainer = SimpleNamespace(device=torch.device("cpu"), ntokens_seen=4)
    provider = TrainerReplayStateProvider(trainer, [module])

    random.seed(7)
    torch.manual_seed(11)
    state = provider.capture()
    expected_python = random.random()
    expected_torch = torch.rand(1)

    module.counter.add_(5)
    module.optional = torch.ones(1)
    module.weight.grad.add_(6)
    trainer.ntokens_seen = 99
    random.random()
    torch.rand(1)

    provider.restore(state)

    torch.testing.assert_close(module.counter, torch.zeros(1))
    assert module.optional is None
    torch.testing.assert_close(module.weight.grad, torch.tensor([3.0]))
    assert trainer.ntokens_seen == 4
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(1), expected_torch)


def test_replay_commits_only_final_candidate():
    module = _BufferModule()
    trainer = SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0)
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True, num_replays=2),
        trainer=trainer,
        modules=[module],
    )
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        module.counter.add_(1)
        trainer.ntokens_seen += 5
        loss = module.weight.square().sum()
        loss.backward()
        return loss

    loss = replay.run(
        execute,
        step=3,
        attempt=1,
    )

    assert calls == 3
    torch.testing.assert_close(loss, torch.tensor(4.0))
    torch.testing.assert_close(module.counter, torch.ones(1))
    torch.testing.assert_close(module.weight.grad, torch.tensor([4.0]))
    assert trainer.ntokens_seen == 5


def test_replay_wraps_compiled_forward_backward():
    module = torch.nn.Linear(4, 2)
    compiled = torch.compile(module, backend="eager")
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True),
        trainer=SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0),
        modules=[module],
    )

    def execute():
        loss = compiled(torch.ones(3, 4)).sum()
        loss.backward()
        return loss

    replay.run(execute, step=1, attempt=1)

    assert module.weight.grad is not None


def test_replay_ignores_unregistered_scratch_state():
    module = _BufferModule()
    trainer = SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0)
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True),
        trainer=trainer,
        modules=[module],
    )
    scratch_invocations = 0

    def execute():
        nonlocal scratch_invocations
        scratch_invocations += 1
        loss = module.weight.square().sum()
        loss.backward()
        return loss

    replay.run(execute, step=1, attempt=1)

    assert scratch_invocations == 2


def test_replay_mismatch_is_fatal():
    module = _BufferModule()
    trainer = SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0)
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True),
        trainer=trainer,
        modules=[module],
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
        replay.run(
            execute,
            step=9,
            attempt=2,
        )


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
    trainer = SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0)
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True),
        trainer=trainer,
        modules=[module],
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
                trainer.ntokens_seen += 1
        return loss

    with pytest.raises(SDCReplayMismatch, match=expected):
        replay.run(execute, step=9, attempt=2)


@pytest.mark.parametrize(
    ("num_steps", "attempt_step", "expected"),
    [(1, 0, True), (1, 1, False), (3, 2, True), (3, 3, False), (-1, 100, True)],
)
def test_attempt_local_scheduling(num_steps, attempt_step, expected):
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True, num_steps=num_steps),
        trainer=SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0),
        modules=[],
    )
    assert replay.should_run(attempt_step) is expected


def test_config_validation():
    with pytest.raises(ValueError, match="num_steps"):
        SDCReplayConfig(num_steps=0).validate()
    with pytest.raises(ValueError, match="num_replays"):
        SDCReplayConfig(num_replays=0).validate()


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
    reference = ReplaySignature(
        schema=schema,
        digests=(
            torch.tensor(1, dtype=torch.uint64),
            torch.tensor(2, dtype=torch.uint64),
        ),
        state=(),
    )
    candidate = ReplaySignature(
        schema=schema,
        digests=(
            torch.tensor(1, dtype=torch.uint64),
            torch.tensor(3, dtype=torch.uint64),
        ),
        state=(),
    )

    with patch("torchtitan.distributed.sdc_replay.torch.equal") as equal:
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
        module.weight.grad = distribute_tensor(torch.tensor([3.0]), mesh, [Replicate()])
        provider = TrainerReplayStateProvider(
            SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0),
            [module],
        )
        state = provider.capture()

        module.counter.to_local().add_(5)
        module.weight.grad.to_local().add_(6)
        provider.restore(state)

        torch.testing.assert_close(module.counter.to_local(), torch.zeros(1))
        torch.testing.assert_close(module.weight.grad.to_local(), torch.tensor([3.0]))
    finally:
        dist.destroy_process_group()


def test_remote_mismatch_is_fatal_on_a_locally_matching_rank():
    replay = SDCReplay(
        config=SDCReplayConfig(enabled=True),
        trainer=SimpleNamespace(device=torch.device("cpu"), ntokens_seen=0),
        modules=[],
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
        signature = ReplaySignature(schema=(), digests=(), state=())
        replay._raise_if_mismatch(
            step=2,
            attempt=1,
            replay=1,
            local_mismatch=torch.tensor(False),
            reference=signature,
            candidate=signature,
        )
