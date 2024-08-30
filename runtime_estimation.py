# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os

import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims
from train import get_train_context
from torchtitan.scyther.runtime_stats import collect_runtime_stats


def estimate_runtime(job_config: JobConfig):
    init_logger()
    logger.info("Estimating runtime...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    # if tp > or pp > 1, we exit
    if (
        job_config.training.tensor_parallel_degree > 1
        or job_config.experimental.pipeline_parallel_degree > 1
    ):
        logger.info(
            "Tensor parallelism and pipeline parallelism are not supported yet."
        )
        return

    # fake tensor doesn't work with fused rmsnorm
    if (
        job_config.model.norm_type == "fused_rmsnorm"
        and not job_config.memory_estimation.disable_fake_mode
    ):
        logger.info(
            "Fused RMSNorm is not supported yet under fake estimation mode. "
            "Switching to rmsnorm."
        )
        job_config.model.norm_type = "rmsnorm"

    if job_config.model.norm_type == "compiled_rmsnorm":
        logger.info("Compiled RMSNorm is not supported yet. Switching to RMSNorm.")
        job_config.model.norm_type = "rmsnorm"

    if job_config.training.compile or job_config.experimental.enable_compiled_autograd:
        logger.info("Compile mode is not supported yet. Switching to eager mode.")
        job_config.training.compile = False
        job_config.experimental.enable_compiled_autograd = False
    
    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1)
        )

    with FakeTensorMode() if job_config.runtime_estimation.estimate_mode_type != "actual" else contextlib.nullcontext():
        device = torch.device(torch.cuda.current_device())
        model_name = job_config.model.name

        # build model (using meta init)
        model_cls = model_name_to_cls[model_name]
        model_config = models_config[model_name][job_config.model.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = job_config.model.norm_type
        model_config.vocab_size = job_config.model.vocab_size
        model_config.max_seq_len = job_config.training.seq_len

        logger.info(
            f"Building {model_name} {job_config.model.flavor} with {model_config}"
        )
        with torch.device(device):
            model = model_cls.from_model_args(model_config)
        model.train()

        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.layers.register_module(layer_id, transformer_block)

        # build optimizer after applying parallelisms to the model
        optimizers = build_optimizers([model], job_config)
        logger.info(f"Vocab size: {model_config.vocab_size}")
        # Create a dummy batch instead of loading from a dataset
        batch = (
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
        )
        collect_runtime_stats(model, optimizers.optimizers[0], batch, loss_fn, job_config.runtime_estimation.estimate_mode_type, job_config)


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    estimate_runtime(config)
