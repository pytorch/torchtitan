# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from lfm2 import LFM2ForCausalLM, LFM2Config

from torchtitan.experiments.lfm2.model.args import LFM2ModelArgs


class LFM2Model(nn.Module):
    """Wrapper around lfm2.LFM2ForCausalLM for TorchTitan integration.

    This wrapper adapts the external lfm2 package to work with TorchTitan's
    training infrastructure.
    """

    def __init__(self, model_args: LFM2ModelArgs) -> None:
        """Initialize LFM2 model.

        Args:
            model_args: LFM2ModelArgs configuration
        """
        super().__init__()
        self.model_args = model_args

        # Create LFM2Config from our model args
        config = LFM2Config(
            vocab_size=model_args.vocab_size,
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            num_conv_blocks=model_args.num_conv_blocks,
            num_attention_blocks=model_args.num_attention_blocks,
            num_attention_heads=model_args.num_attention_heads,
            num_key_value_heads=model_args.num_key_value_heads,
            conv_kernel_size=model_args.conv_kernel_size,
            max_position_embeddings=model_args.max_position_embeddings,
            rms_norm_eps=model_args.rms_norm_eps,
            tie_word_embeddings=model_args.tie_word_embeddings,
            rope_theta=model_args.rope_theta,
            attention_dropout=model_args.attention_dropout,
            hidden_dropout=model_args.hidden_dropout,
            initializer_range=model_args.initializer_range,
            pad_token_id=model_args.pad_token_id,
            bos_token_id=model_args.bos_token_id,
            eos_token_id=model_args.eos_token_id,
            use_cache=False,  # Don't use KV cache during training
            use_return_dict=True,
        )

        # Create the actual LFM2 model
        self.model = LFM2ForCausalLM(config)

    @property
    def layers(self):
        """Expose layers for activation checkpointing.

        Returns the ModuleList of layers from the inner LFM2Model.
        """
        return self.model.model.layers

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights.

        This properly initializes weights when the model is created on meta device
        and then moved to a real device.

        Args:
            buffer_device: Optional device to place buffers on during initialization.
        """
        def _init_weights(module):
            """Initialize weights for different module types."""
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.model_args.initializer_range,
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.model_args.initializer_range,
                )
            elif isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.model_args.initializer_range,
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            # RMSNorm weights should be initialized to ones
            elif module.__class__.__name__ == 'RMSNorm':
                if hasattr(module, 'weight'):
                    torch.nn.init.ones_(module.weight)

        # Apply initialization to all modules
        self.model.apply(_init_weights)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            **kwargs: Additional arguments (labels, attention_mask, etc.)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        outputs = self.model(input_ids=input_ids, return_dict=True, **kwargs)
        return outputs["logits"]

    def reset_parameters(self) -> None:
        """Reset model parameters.

        This re-initializes the model weights using the lfm2 package's
        initialization strategy.
        """
        self.model.apply(self.model._init_weights)
