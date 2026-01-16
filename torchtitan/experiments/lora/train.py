# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LoRA Training Entry Point

This module provides a training entry point that enables LoRA (Low-Rank Adaptation)
for fine-tuning large language models. It imports the LoRA module to register the
LoRAConverter, then delegates to the main training logic.

Usage:
    Run training with LoRA enabled:
    ```
    CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" \\
    ./run_train.sh --training.steps 10 torchtitan.experiments.lora.train
    ```

    Make sure to add "lora" to the model.converters list in your config:
    ```toml
    [model]
    converters = ["lora"]
    ```
"""

# Import LoRA module to register the LoRAConverter with the model converter registry
import torchtitan.experiments.lora.lora  # noqa: F401

from torchtitan.train import main, Trainer


if __name__ == "__main__":
    main(Trainer)
