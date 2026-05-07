# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.comm_buffer_lifetime import (
    AsyncAllGatherResult,
    AsyncReduceScatterResult,
    StreamHandoff,
)
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cpu_mesh,
    transformer_bucket_specs,
    transformer_inputs,
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
    def test_eager_forward_backward_on_cpu_mesh(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            loss = model(transformer_inputs(args)).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_param_access_outside_forward_raises(self):
        with single_rank_cpu_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
                _ = model.output.weight

    def test_graph_capture_raises(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(transformer_inputs(args))

    def test_graph_capture_error_does_not_poison_next_eager_forward(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)
            inp = transformer_inputs(args)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(inp)

            loss = model(inp).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

    def test_graph_capture_raises_before_collectives(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            with (
                patch.object(torch.compiler, "is_compiling", return_value=True),
                patch.object(
                    Shard,
                    "unshard",
                    side_effect=AssertionError("unshard should not run"),
                ) as unshard,
                patch.object(
                    Shard,
                    "begin_unshard",
                    side_effect=AssertionError("begin_unshard should not run"),
                ) as begin_unshard,
            ):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(transformer_inputs(args))

            unshard.assert_not_called()
            begin_unshard.assert_not_called()

    def test_cpu_runtime_runs_one_unshard_and_reduce_per_bucket(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = make_transformer_model()
            flex_shard_cpu(
                model,
                mesh,
                buckets=transformer_bucket_specs(
                    args.n_layers,
                    reshard_after_forward=False,
                ),
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
                model(transformer_inputs(args)).sum().backward()

            expected_calls = [list(storage.param_infos) for storage in model.dstorages]
            self.assertEqual(unshard_calls, expected_calls)
            self.assertCountEqual(
                [tuple(call) for call in reduce_calls],
                [tuple(call) for call in expected_calls],
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
