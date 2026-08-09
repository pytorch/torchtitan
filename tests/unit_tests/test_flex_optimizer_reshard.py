# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch
from torchtitan.components.distributed_optimizers.flex_optimizer_reshard import (
    _BucketedRedistributionRuntime,
)


class TestFlexOptimizerReshard(unittest.TestCase):
    def test_runtime_enqueues_return_before_next_compute(self):
        runtime = _BucketedRedistributionRuntime(torch.device("cuda"))
        caller_stream = Mock()
        context = Mock()
        context.device_handle.current_stream.return_value = caller_stream
        context.device_handle.stream.side_effect = lambda _stream: nullcontext()
        context.device_handle.Event.side_effect = Mock
        context.slots = (Mock(), Mock())
        for slot in context.slots:
            slot.communication_buffers.return_value = (object(), object())
        runtime._context = context

        plans = tuple(
            Mock(redistributed_items=(object(),), unredistributed_items=())
            for _ in range(3)
        )
        plan_names = {plan: f"bucket_{index}" for index, plan in enumerate(plans)}
        events = []
        for plan in plans:
            plan.storage_to_compute_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("gather", plan_names[_plan])
                )
            )
            plan.compute_to_storage_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("return", plan_names[_plan])
                )
            )

        def compute_redistributed(work, *_args, **_kwargs):
            events.append(("compute", plan_names[work.plan]))

        original_release = runtime._release

        def release(work, caller):
            events.append(("release", plan_names[work.plan]))
            original_release(work, caller)

        with patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._prepare_redistributed"
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._compute_redistributed",
            side_effect=compute_redistributed,
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._finalize_redistributed"
        ), patch.object(
            runtime,
            "_release",
            side_effect=release,
        ):
            runtime.run(
                plans,
                local_tensor_spec=Mock(),
                prepare=Mock(),
                compute=Mock(),
                finalize=Mock(),
            )

        self.assertEqual(
            events,
            [
                ("gather", "bucket_0"),
                ("compute", "bucket_0"),
                ("gather", "bucket_1"),
                ("return", "bucket_0"),
                ("compute", "bucket_1"),
                ("release", "bucket_0"),
                ("gather", "bucket_2"),
                ("return", "bucket_1"),
                ("compute", "bucket_2"),
                ("release", "bucket_1"),
                ("return", "bucket_2"),
                ("release", "bucket_2"),
            ],
        )
        self.assertEqual(context.slots[0].communication_buffers.call_count, 2)
        self.assertEqual(context.slots[1].communication_buffers.call_count, 1)
        context.transfer_stream.wait_stream.assert_called_once_with(caller_stream)
        caller_stream.wait_stream.assert_not_called()

    def test_runtime_orders_streams_when_finalize_raises(self):
        runtime = _BucketedRedistributionRuntime(torch.device("cuda"))
        caller_stream = Mock()
        transfer_stream = Mock()
        context = Mock()
        context.transfer_stream = transfer_stream
        context.device_handle.current_stream.return_value = caller_stream
        context.device_handle.stream.side_effect = lambda _stream: nullcontext()
        context.device_handle.Event.side_effect = Mock
        context.slots = (Mock(), Mock())
        context.slots[0].communication_buffers.return_value = (object(), object())
        runtime._context = context

        plan = Mock(redistributed_items=(object(),), unredistributed_items=())
        events = []
        transfer_stream.wait_stream.side_effect = lambda stream: events.append(
            ("transfer_wait", stream)
        )
        caller_stream.wait_stream.side_effect = lambda stream: events.append(
            ("caller_wait", stream)
        )
        plan.storage_to_compute_schedule.execute.side_effect = (
            lambda **_kwargs: events.append(("gather",))
        )
        plan.compute_to_storage_schedule.execute.side_effect = (
            lambda **_kwargs: events.append(("return",))
        )
        failure = RuntimeError("finalize failed")

        with patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._prepare_redistributed"
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._compute_redistributed"
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._finalize_redistributed",
            side_effect=failure,
        ):
            with self.assertRaises(RuntimeError) as raised:
                runtime.run(
                    (plan,),
                    local_tensor_spec=Mock(),
                    prepare=Mock(),
                    compute=Mock(),
                    finalize=Mock(),
                )

        self.assertIs(raised.exception, failure)
        self.assertEqual(
            events,
            [
                ("transfer_wait", caller_stream),
                ("gather",),
                ("return",),
                ("transfer_wait", caller_stream),
                ("caller_wait", transfer_stream),
            ],
        )


if __name__ == "__main__":
    unittest.main()
