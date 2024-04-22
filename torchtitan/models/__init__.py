# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import llama_configs, Transformer

models_config = {
    "llama": llama_configs,
}

model_name_to_cls = {
    "llama": Transformer,
}

model_name_to_tokenizer = {
    "llama": "sentencepiece",
}
