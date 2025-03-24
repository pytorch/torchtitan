
#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NGPU=${NGPU:-"4"}

# Get the prompt from command line argument or use a default
prompt="${1:-What is 2+2?}"

# Run the model with the prompt
torchrun --standalone --nproc-per-node ${NGPU} generate.py "$prompt"
