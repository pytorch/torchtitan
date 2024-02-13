# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from torchtrain.logging_utils import rank0_log
from torchtrain.profiling import get_config_from_toml


class MetricLogger:
    def __init__(self, log_dir, tag, enable_tb):
        self.tag = tag
        self.writer: Optional[SummaryWriter] = None
        if enable_tb:
            self.writer = SummaryWriter(log_dir, max_queue=1000)

    def log(self, metrics: Dict[str, Any], step: int):
        if self.writer is not None:
            for k, v in metrics.items():
                tag = k if self.tag is None else f"{self.tag}/{k}"
                self.writer.add_scalar(tag, v, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def build_metric_logger(tag: Optional[str] = None):
    config = get_config_from_toml()

    dump_dir = config["global"]["dump_folder"]
    save_tb_folder = config["metrics"]["save_tb_folder"]
    # since we don't have run id yet, use current minute as identifier
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, save_tb_folder, datetime_str)

    enable_tb = config["metrics"].get("enable_tensorboard", False)
    if enable_tb:
        rank0_log(
            f"Metrics logging active. Tensorboard logs will be saved at {log_dir}."
        )

    rank_str = f"rank_{torch.distributed.get_rank()}"
    return MetricLogger(os.path.join(log_dir, rank_str), tag, enable_tb)
