# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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
