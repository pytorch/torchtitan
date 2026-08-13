# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full configurations backing the integration tests.

Each function here is one entry in ``tests/integration_tests``, expressed as
a configuration instead of a base config plus command-line flags. Keeping
them in this package rather than in the test files means CI exercises the
same selection path users do.

Unrelated to the repository's top-level ``tests/`` package, which holds the
test code itself.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import model_registry
from torchtitan.trainer import Trainer


def llama3_debugmodel_fsdp2_cp2() -> Trainer.Config:
    """Debug model on 4 GPUs: FSDP 2, context parallel 2.

    Pins the parallelism the run needs instead of leaving it to the command
    line, so the configuration name is enough to reproduce it. Exercised by
    the ``fsdp+cp`` integration test.
    """
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        model_spec=model_spec,
        hf_assets_path="./tests/assets/tokenizer",
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        optimizer=default_adamw(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            context_parallel_degree=2,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        activation_checkpoint=SelectiveAC.Config(),
        checkpoint=CheckpointManager.Config(interval=10),
        metrics=MetricsProcessor.Config(log_freq=1),
        validator=Validator.Config(freq=5, steps=10),
    )
