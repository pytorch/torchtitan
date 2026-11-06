# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

from torchtitan.observability.metrics import TensorBoardLogger


def test_every_scalar_readable_after_close_one_event_per_step(tmp_path):
    tb_logger = TensorBoardLogger.Config().build(log_dir=str(tmp_path), tag="train")
    # The events file is our unbuffered handle, not TensorBoard's per-event GFile.
    events_handle = tb_logger.writer.event_writer._async_writer._writer._writer
    assert isinstance(events_handle, io.FileIO)
    for step in range(3):
        # "note" can't be a float, so it is skipped and the step keeps the other scalars.
        tb_logger.log({"loss": 1.0 + step, "lr": 0.5 * step, "note": None}, step)
    tb_logger.close()

    accumulator = EventAccumulator(str(tmp_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    assert {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    } == {
        "train/loss": [(0, 1.0), (1, 2.0), (2, 3.0)],
        "train/lr": [(0, 0.0), (1, 0.5), (2, 1.0)],
    }
    (events_file,) = tmp_path.glob("events.out.tfevents.*")
    events = EventFileLoader(str(events_file)).Load()
    assert [event.step for event in events if event.HasField("summary")] == [0, 1, 2]
