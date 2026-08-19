# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchtitan.components.metrics import TensorBoardLogger


def test_tensorboard_logger_writes_batched_scalars(tmp_path) -> None:
    logger = TensorBoardLogger(str(tmp_path), tag="train")
    file_writer = logger.writer._get_file_writer()
    with patch.object(
        file_writer,
        "add_summary",
        wraps=file_writer.add_summary,
    ) as add_summary:
        logger.log(
            {"loss": 1.25, "num_tokens": 8, "grad_norm": torch.tensor(-2.5)},
            step=17,
        )

    add_summary.assert_called_once()
    summary, step = add_summary.call_args.args
    assert step == 17
    assert {value.tag for value in summary.value} == {
        "train/loss",
        "train/num_tokens",
        "train/grad_norm",
    }
    logger.close()

    event_accumulator = EventAccumulator(str(tmp_path)).Reload()
    assert set(event_accumulator.Tags()["scalars"]) == {
        "train/loss",
        "train/num_tokens",
        "train/grad_norm",
    }
    assert {
        tag: [(event.step, event.value) for event in event_accumulator.Scalars(tag)]
        for tag in event_accumulator.Tags()["scalars"]
    } == {
        "train/loss": [(17, 1.25)],
        "train/num_tokens": [(17, 8.0)],
        "train/grad_norm": [(17, -2.5)],
    }
