# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.comm_buffer_lifetime import (
    AsyncAllGatherResult,
    AsyncReduceScatterResult,
    StreamHandoff,
)
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    single_rank_cpu_mesh,
)


class _FakeStream:
    def __init__(self) -> None:
        self.waited_events = []

    def wait_event(self, event) -> None:
        self.waited_events.append(event)


class _FakeDeviceHandle:
    def __init__(self) -> None:
        self.streams = []

    def stream(self, stream):
        self.streams.append(stream)
        return nullcontext()


class TestFlexShardEagerRuntime(unittest.TestCase):
    def test_cpu_runtime_runs_one_unshard_and_reduce_per_bucket(self):
        with single_rank_cpu_mesh() as mesh:
            model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
            flex_shard_cpu(
                model,
                mesh,
                buckets=[
                    BucketSpec(["0.*"], reshard_after_forward=False),
                    BucketSpec(["2.*"], reshard_after_forward=False),
                ],
            )
            unshard_calls = []
            reduce_calls = []
            original_unshard = Shard.unshard
            original_reduce_grad = Shard.reduce_grad

            def _count_unshard(tensors, infos, mesh):
                unshard_calls.append([info.fqn for info in infos])
                return original_unshard(tensors, infos, mesh)

            def _count_reduce_grad(tensors, infos, mesh):
                reduce_calls.append([info.fqn for info in infos])
                return original_reduce_grad(tensors, infos, mesh)

            with (
                patch.object(Shard, "unshard", side_effect=_count_unshard),
                patch.object(
                    Shard,
                    "reduce_grad",
                    side_effect=_count_reduce_grad,
                ),
            ):
                model(torch.randn(3, 4)).sum().backward()

            self.assertEqual(
                unshard_calls,
                [["0.weight", "0.bias"], ["2.weight", "2.bias"]],
            )
            self.assertEqual(
                reduce_calls,
                [["2.weight", "2.bias"], ["0.weight", "0.bias"]],
            )

    def test_release_unshard_buffers_clears_cpu_tensors(self):
        result = AsyncAllGatherResult(
            gathered=[torch.ones(1), torch.ones(2)],
            infos=[],
            mesh=None,
            per_rank_param_offsets=[],
            event=None,
            send_buf=torch.ones(3),
            debug_fqn=None,
        )

        Shard.release_unshard_buffers(result)

        self.assertEqual(result.gathered, [])
        self.assertIsNone(result.send_buf)

    def test_release_reduce_grad_buffers_clears_cpu_tensors(self):
        result = AsyncReduceScatterResult(
            sharded_grads=[torch.ones(1), torch.ones(2)],
            event=None,
            send_buf=torch.ones(3),
            recv_buf=torch.ones(3),
            debug_fqn=None,
        )

        Shard.release_reduce_grad_buffers(result, release_sharded_grads=True)

        self.assertEqual(result.sharded_grads, [])
        self.assertIsNone(result.send_buf)
        self.assertIsNone(result.recv_buf)

    def test_stream_handoff_wait_and_release_are_idempotent(self):
        event = object()
        wait_stream = _FakeStream()
        release_stream = _FakeStream()
        device_handle = _FakeDeviceHandle()
        handoff = StreamHandoff(
            torch.ones(1),
            event,
            release_stream,
            device_handle=device_handle,
        )

        handoff.wait(wait_stream)
        handoff.release()
        handoff.release()

        self.assertEqual(wait_stream.waited_events, [event])
        self.assertEqual(release_stream.waited_events, [event])
        self.assertEqual(device_handle.streams, [release_stream])
        self.assertIsNone(handoff._tensor)


if __name__ == "__main__":
    unittest.main()
