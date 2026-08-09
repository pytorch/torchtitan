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
    _PipelineSlot,
    _RedistributionBucketPlan,
)


class TestFlexOptimizerReshard(unittest.TestCase):
    def test_runtime_enqueues_return_before_next_compute(self):
        runtime = _BucketedRedistributionRuntime(
            torch.device("cuda"),
            num_pipeline_slots=3,
        )
        caller_stream = Mock()
        transfer_stream = Mock()
        device_handle = Mock()
        device_handle.current_stream.return_value = caller_stream
        device_handle.Stream.return_value = transfer_stream
        device_handle.stream.side_effect = lambda _stream: nullcontext()
        events_by_slot = tuple(tuple(Mock() for _ in range(3)) for _ in range(3))
        device_handle.Event.side_effect = (
            event for slot_events in events_by_slot for event in slot_events
        )

        plans = []
        for size in (4, 3, 8, 6, 10):
            storage_to_compute = Mock(
                input_buffer_numel=size,
                output_buffer_numel=size + 1,
            )
            compute_to_storage = Mock(
                input_buffer_numel=size + 3,
                output_buffer_numel=size + 2,
            )
            redistribution_plan = Mock()
            redistribution_plan.storage_partition.return_value = Mock(
                tensor_shape=(size,)
            )
            redistribution_plan.compute_partition.return_value = Mock(
                tensor_shape=(size,)
            )
            plans.append(
                _RedistributionBucketPlan(
                    redistributed_items=(object(),),
                    unredistributed_items=(),
                    redistribution_plans=(redistribution_plan,),
                    group=Mock(local_participant=0),
                    storage_to_compute_schedule=storage_to_compute,
                    compute_to_storage_schedule=compute_to_storage,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            )
        plans = tuple(plans)
        with patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard.torch.get_device_module",
            return_value=device_handle,
        ):
            runtime.reserve_buffers(plans, local_tensor_spec=Mock())
        context = runtime._context
        self.assertIsNotNone(context)
        self.assertEqual(device_handle.Event.call_count, 9)
        for slot, (input_ready, compute_done, done) in zip(
            context.slots, events_by_slot, strict=True
        ):
            self.assertIs(slot.compute_input_ready, input_ready)
            self.assertIs(slot.compute_done, compute_done)
            self.assertIs(slot.done, done)
            input_ready.record.assert_called_once_with(transfer_stream)
            compute_done.record.assert_called_once_with(caller_stream)
            done.record.assert_called_once_with(transfer_stream)
            input_ready.reset_mock()
            compute_done.reset_mock()
            done.reset_mock()
        device_handle.Event.reset_mock()

        key = (torch.device("cpu"), torch.float32)
        slot_0_buffers = context.slots[0].buffers.buffers[key]
        slot_1_buffers = context.slots[1].buffers.buffers[key]
        slot_2_buffers = context.slots[2].buffers.buffers[key]
        self.assertEqual(slot_0_buffers.storage_exchange.numel(), 8)
        self.assertEqual(slot_0_buffers.compute_exchange.numel(), 9)
        self.assertEqual(slot_0_buffers.compute_scratch.numel(), 6)
        self.assertEqual(slot_0_buffers.storage_scratch.numel(), 6)
        self.assertEqual(slot_1_buffers.storage_exchange.numel(), 12)
        self.assertEqual(slot_1_buffers.compute_exchange.numel(), 13)
        self.assertEqual(slot_1_buffers.compute_scratch.numel(), 10)
        self.assertEqual(slot_1_buffers.storage_scratch.numel(), 10)
        self.assertEqual(slot_2_buffers.storage_exchange.numel(), 10)
        self.assertEqual(slot_2_buffers.compute_exchange.numel(), 11)
        self.assertEqual(slot_2_buffers.compute_scratch.numel(), 8)
        self.assertEqual(slot_2_buffers.storage_scratch.numel(), 8)
        self.assertNotEqual(
            slot_0_buffers.compute_scratch.data_ptr(),
            slot_0_buffers.storage_scratch.data_ptr(),
        )
        self.assertNotEqual(
            slot_0_buffers.storage_exchange.data_ptr(),
            slot_1_buffers.storage_exchange.data_ptr(),
        )
        plan_names = {id(plan): f"bucket_{index}" for index, plan in enumerate(plans)}
        events = []
        for plan in plans:
            plan.storage_to_compute_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("gather", plan_names[id(_plan)])
                )
            )
            plan.compute_to_storage_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("return", plan_names[id(_plan)])
                )
            )

        def compute_redistributed(work, *_args, **_kwargs):
            events.append(("compute", plan_names[id(work.plan)]))

        original_release = runtime._release

        def release(work, caller):
            events.append(("release", plan_names[id(work.plan)]))
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
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._reserve_tensor"
        ) as reserve_tensor, patch.object(
            runtime,
            "_release",
            side_effect=release,
        ), patch.object(
            context.device_handle,
            "Event",
        ) as create_event:
            runtime.run(
                plans,
                local_tensor_spec=Mock(),
                prepare=Mock(),
                compute=Mock(),
                finalize=Mock(),
            )
        reserve_tensor.assert_not_called()
        create_event.assert_not_called()

        self.assertEqual(
            events,
            [
                ("gather", "bucket_0"),
                ("compute", "bucket_0"),
                ("gather", "bucket_1"),
                ("gather", "bucket_2"),
                ("return", "bucket_0"),
                ("compute", "bucket_1"),
                ("release", "bucket_0"),
                ("gather", "bucket_3"),
                ("return", "bucket_1"),
                ("compute", "bucket_2"),
                ("release", "bucket_1"),
                ("gather", "bucket_4"),
                ("return", "bucket_2"),
                ("compute", "bucket_3"),
                ("release", "bucket_2"),
                ("return", "bucket_3"),
                ("compute", "bucket_4"),
                ("release", "bucket_3"),
                ("return", "bucket_4"),
                ("release", "bucket_4"),
            ],
        )
        self.assertEqual(context.slots[0].compute_input_ready.record.call_count, 2)
        self.assertEqual(context.slots[1].compute_input_ready.record.call_count, 2)
        self.assertEqual(context.slots[2].compute_input_ready.record.call_count, 1)
        self.assertEqual(context.slots[0].done.record.call_count, 2)
        self.assertEqual(context.slots[1].done.record.call_count, 2)
        self.assertEqual(context.slots[2].done.record.call_count, 1)
        context.transfer_stream.wait_stream.assert_called_once_with(caller_stream)
        caller_stream.wait_stream.assert_not_called()

    def test_runtime_orders_streams_when_finalize_raises(self):
        runtime = _BucketedRedistributionRuntime(
            torch.device("cuda"),
            num_pipeline_slots=3,
        )
        caller_stream = Mock()
        transfer_stream = Mock()
        context = Mock()
        context.transfer_stream = transfer_stream
        context.device_handle.current_stream.return_value = caller_stream
        context.device_handle.stream.side_effect = lambda _stream: nullcontext()
        context.slots = tuple(
            _PipelineSlot(
                buffers=Mock(),
                compute_input_ready=Mock(),
                compute_done=Mock(),
                done=Mock(),
            )
            for _ in range(3)
        )
        for slot in context.slots:
            slot.buffers.communication_buffers.return_value = (object(), object())
        runtime._context = context

        plans = tuple(
            Mock(redistributed_items=(object(),), unredistributed_items=())
            for _ in range(3)
        )
        plan_names = {plan: index for index, plan in enumerate(plans)}
        events = []
        transfer_stream.wait_stream.side_effect = lambda stream: events.append(
            ("transfer_wait", stream)
        )
        caller_stream.wait_stream.side_effect = lambda stream: events.append(
            ("caller_wait", stream)
        )
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
        failure = RuntimeError("finalize failed")

        with patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._prepare_redistributed"
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._compute_redistributed",
            side_effect=lambda work, *_args, **_kwargs: events.append(
                ("compute", plan_names[work.plan])
            ),
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._finalize_redistributed",
            side_effect=failure,
        ):
            with self.assertRaises(RuntimeError) as raised:
                runtime.run(
                    plans,
                    local_tensor_spec=Mock(),
                    prepare=Mock(),
                    compute=Mock(),
                    finalize=Mock(),
                )

        self.assertIs(raised.exception, failure)
        context.device_handle.Event.assert_not_called()
        self.assertEqual(
            events,
            [
                ("transfer_wait", caller_stream),
                ("gather", 0),
                ("compute", 0),
                ("gather", 1),
                ("gather", 2),
                ("return", 0),
                ("transfer_wait", caller_stream),
                ("caller_wait", transfer_stream),
            ],
        )


if __name__ == "__main__":
    unittest.main()
