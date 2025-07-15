# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import mlperf_logging.mllog as mllog

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import constants, MLLoggerWrapper

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)


class DeltaTimer:
    def __init__(self):
        self.reset()
        self.start_time = None

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


class MLPerfLogger:
    def __init__(
        self,
        root_dir: str,
        filename: str | None = None,
        default_stack_offset: int = 2,
        log_every_n_steps: int = 10,
    ):
        self.mllogger = mllogger
        self.log_every_n_steps = log_every_n_steps
        self.timer = DeltaTimer()
        mllog.config(
            default_stack_offset=default_stack_offset,
            filename=filename
            or os.getenv("COMPLIANCE_FILE")
            or "mlperf_compliance.log",
            root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))),
        )

        self.submission_info = {
            constants.SUBMISSION_BENCHMARK: "flux",
            constants.SUBMISSION_DIVISION: "closed",
            constants.SUBMISSION_ORG: "reference",
            constants.SUBMISSION_PLATFORM: "reference",
            constants.SUBMISSION_POC_NAME: "reference",
            constants.SUBMISSION_POC_EMAIL: "reference",
            constants.SUBMISSION_STATUS: "onprem",
            constants.TRAIN_SAMPLES: 1_099_776,
            constants.EVAL_SAMPLES: 29_696,
        }

    def log_run_start(
        self,
        gbs: int,
        seed: int,
        lr: float,
        warmup_steps: int,
        gradient_clip_norm: float,
        optimizer_config: dict,
        eval_freq: int,
    ):
        self.gbs = gbs
        self.mllogger.event(
            key=constants.CACHE_CLEAR,
            value="True",
        )
        for key, value in self.submission_info.items():
            self.mllogger.event(key=key, value=value)

        self.mllogger.event(key=constants.SEED, value=seed)
        self.mllogger.event(key=constants.GLOBAL_BATCH_SIZE, value=gbs)

        self.mllogger.event(key=constants.OPT_NAME, value=constants.ADAMW)
        self.mllogger.event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_steps)
        self.mllogger.event(
            key=constants.OPT_ADAMW_BETA_1, value=optimizer_config["betas"][0]
        )
        self.mllogger.event(
            key=constants.OPT_ADAMW_BETA_2, value=optimizer_config["betas"][1]
        )
        self.mllogger.event(
            key=constants.OPT_ADAMW_EPSILON, value=optimizer_config["eps"]
        )
        self.mllogger.event(
            key=constants.OPT_ADAMW_WEIGHT_DECAY, value=optimizer_config["weight_decay"]
        )
        self.mllogger.event(key=constants.OPT_BASE_LR, value=lr)
        self.mllogger.event(
            key=constants.OPT_GRADIENT_CLIP_NORM, value=gradient_clip_norm
        )
        self.mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=1)
        self.mllogger.event(key="evaluation_frequency", value=eval_freq)
        self.mllogger.start(key=constants.INIT_START, value="")

    def log_train_start(self):
        if self.timer.start_time is None:
            self.timer.reset()
        self.mllogger.log_init_stop_run_start()

    def log_train_end(self, success: bool):
        self.mllogger.log_run_stop(
            status=constants.SUCCESS if success else constants.ABORTED,
        )

    def log_train_step_start(self, step: int):
        if (step - 1) % self.log_every_n_steps != 0:
            return
        self.mllogger.start(
            key=constants.BLOCK_START,
            value="training_step",
            metadata={constants.SAMPLES_COUNT: (step - 1) * self.gbs},
        )

    def log_train_step_end(self, step: int, loss: float, lr: float):
        if (step - 1) % self.log_every_n_steps != 0:
            return

        block_time = self.timer.get_delta()
        throughput = self.log_every_n_steps * self.gbs / block_time
        self.mllogger.event(
            key="tracked_stats",
            value={
                "throughput": throughput,
                "loss": loss,
                "lr": lr,
                "train_step_time": block_time / self.log_every_n_steps,
            },
            metadata={constants.STEP_NUM: step - 1},
        )
        self.mllogger.end(
            key=constants.BLOCK_STOP,
            value="training_step",
            metadata={constants.SAMPLES_COUNT: (step - 1) * self.gbs},
        )

    def log_eval_start(self, step: int):
        self.mllogger.event(
            key=constants.EVAL_START,
            metadata={constants.SAMPLES_COUNT: step * self.gbs},
        )

    def log_eval_end(self, step: int, loss: float):
        self.mllogger.event(
            key=constants.EVAL_ACCURACY,
            value=loss,
            metadata={constants.SAMPLES_COUNT: step * self.gbs},
        )
        self.mllogger.end(
            key=constants.EVAL_STOP,
            value=loss,
            metadata={constants.SAMPLES_COUNT: step * self.gbs},
        )
