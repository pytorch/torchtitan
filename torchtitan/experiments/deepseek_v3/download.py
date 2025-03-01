# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
# python download.py {model_id}
# Example:
# python download.py deepseek-ai/DeepSeek-V2-Lite

import sys

from transformers import AutoModelForCausalLM

model_id = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
