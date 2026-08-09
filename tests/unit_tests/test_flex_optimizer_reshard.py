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
)


class TestFlexOptimizerReshard(unittest.TestCase):
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
