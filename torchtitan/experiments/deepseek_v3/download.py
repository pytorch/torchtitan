# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
# Downloads a given model to the HF Cache.  Pass in a listed option ala "v3" or your own custom model path.
# python download.py {model_id} [custom_model_path]
# Examples:
# python download.py v2     # Use predefined model: deepseek-ai/DeepSeek-V2
# python download.py custom "deepseek-ai/new-model"  # Download a custom model path

# Available models:
#   "v2-lite-chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
#   "v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
#   "v2": "deepseek-ai/DeepSeek-V2",
#   "v3": "deepseek-ai/deepseek-v3",
#   "v3-0324": "deepseek-ai/DeepSeek-V3-0324",
#   "custom": None,  # Placeholder for custom models


import sys

from transformers import AutoModelForCausalLM


MODELS = {
    "v2-lite-chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "v2": "deepseek-ai/DeepSeek-V2",
    "v3": "deepseek-ai/deepseek-v3",
    "v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "custom": None,  # For custom (any) models
}


def print_usage():
    print("Usage:")
    print("  python download.py [model_version]")
    print("  python download.py custom [custom_model_path]")
    print("\nAvailable predefined models:")
    for key, model in MODELS.items():
        if key != "custom":  # Skip the custom placeholder
            print(f"  {key}: {model}")
    print("\nFor custom models:")
    print("  custom: Specify your own model path")
    print('  Example: python download.py custom "organization/model-name"')
    sys.exit(1)


# Process command line arguments
if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
    print_usage()

if sys.argv[1] == "custom":
    if len(sys.argv) != 3:
        print("Error: Custom model requires a model path")
        print_usage()
    model_id = sys.argv[2]
    print(f"Using custom model: {model_id}")
else:
    model_id = MODELS[sys.argv[1]]
print(f"Downloading model: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)

print(f"{model=}")
