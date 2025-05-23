# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch import nn, Tensor
from transformers import CLIPTextModel, T5EncoderModel


class FluxEmbedder(nn.Module):
    def __init__(self, version: str, random_init=False, **hf_kwargs):
        super().__init__()
        self.is_clip = "clip" in version.lower()
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        if self.is_clip:
            if random_init:
                # Initialize CLIP model with random weights for test purpose only
                self.hf_module = CLIPTextModel._from_config(
                    CLIPTextModel.config_class.from_pretrained(
                        os.path.join(version, "config.json"), **hf_kwargs
                    )
                )
            else:
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                    version, **hf_kwargs
                )
        else:
            if random_init:
                # Initialize T5 model with random weights for test purpose only
                self.hf_module = T5EncoderModel._from_config(
                    T5EncoderModel.config_class.from_pretrained(
                        os.path.join(version, "config.json"), **hf_kwargs
                    )
                )
            else:
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                    version, **hf_kwargs
                )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, batch_tokens: Tensor) -> Tensor:
        """
        batch_tokens: [bsz, embedding_length]

        For T5 Encoder, embeding_length is 768
        For CLIP, embedding_length is 256
        """
        outputs = self.hf_module(
            input_ids=batch_tokens.to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
