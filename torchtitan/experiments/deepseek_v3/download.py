# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
# python download.py {model_id}
# Example:
# python download.py v2   # v2 = deepseek-ai/DeepSeek-V2-Lite-Chat

# Available models:
# "deepseek-ai/DeepSeek-V2-Lite"
# "deepseek-ai/deepseek-v3"

# Note - Trust remote code is set to yes to download the model.

import sys

from transformers import AutoModelForCausalLM


MODELS = {"v2": "deepseek-ai/DeepSeek-V2-Lite-Chat", "v3": "deepseek-ai/deepseek-v3"}

if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
    print("Usage: python download.py [model_version]")
    print("Available models:")
    for key, model in MODELS.items():
        print(f"{key}: {model}")
    sys.exit(1)

model_id = MODELS[sys.argv[1]]
print(f"Downloading model: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)
