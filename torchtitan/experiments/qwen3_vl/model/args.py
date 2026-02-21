# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs


@dataclass
class SpecialTokens:
    """Special tokens for Qwen3-VL multimodal processing."""

    img_token: str
    img_id: int
    vision_start_token: str
    vision_start_id: int
    vision_end_token: str
    vision_end_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # PyTorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        SPECIAL_TOKENS_MAP = {
            "img": "<|image_pad|>",
            "vision_start": "<|vision_start|>",
            "vision_end": "<|vision_end|>",
            "pad": "<|endoftext|>",
        }
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        # Try to get tokens from added tokens, fall back to encode if not found
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            if tok in token_to_id:
                special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
            else:
                # Fall back to encoding the token
                encoded = tokenizer.encode(tok)
                if len(encoded) != 1:
                    raise ValueError(
                        f"Special token '{tok}' encodes to {len(encoded)} tokens "
                        f"but must encode to exactly 1 token. "
                        f"Please use a tokenizer with Qwen3-VL special tokens "
                        f"(e.g., Qwen/Qwen3-VL-8B-Instruct)."
                    )
                special_tokens_dict[f"{prefix}_id"] = encoded[0]
        return cls(**special_tokens_dict)


@dataclass
class Qwen3VLVisionEncoderArgs:
    """Arguments for Qwen3-VL Vision Encoder (ViT)."""

    # Transformer dimensions
    dim: int = 1280  # Hidden size of the vision encoder
    ffn_dim: int = 5120  # Intermediate size (4 * dim typically)
    n_layers: int = 32  # Number of transformer layers (depth)
    n_heads: int = 16  # Number of attention heads

    # Vision-specific parameters
    patch_size: int = 14  # Size of each image patch
    temporal_patch_size: int = 2  # Temporal patch size for video
    in_channels: int = 3  # Number of input channels (RGB)
    spatial_merge_size: int = 2  # Merge spatial patches to reduce sequence length

    # Output dimension (maps to LLM hidden size)
    out_hidden_size: int = 3584  # Output hidden size after merging

    # Position embeddings
    num_position_embeddings: int = 4096  # Number of learnable position embeddings (64x64 grid)

    # Normalization and attention
    layer_norm_eps: float = 1e-6

    # RoPE parameters for positional encoding
    rope_theta: float = 10000.0

    # DeepStack: layer indices for extracting intermediate visual features
    # These features are added to the LLM hidden states at early layers
    deepstack_visual_indicies: list[int] = field(default_factory=lambda: [7, 15, 23])


@dataclass
class Qwen3VLTextConfig:
    """Text/LLM configuration with MRoPE support."""

    # MRoPE section sizes for interleaved multi-dimensional RoPE
    # [temporal, height, width] - controls how position dimensions are interleaved
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])


@dataclass
class Qwen3VLModelArgs(Qwen3ModelArgs):
    """Arguments for the complete Qwen3-VL model."""

    # Vision encoder configuration
    encoder: Qwen3VLVisionEncoderArgs = field(default_factory=Qwen3VLVisionEncoderArgs)

    # Text/LLM configuration extensions
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)

    # Vision-language specific token IDs
    image_token_id: int = 151655  # Token ID for image placeholder
    video_token_id: int = 151656  # Token ID for video placeholder
    vision_start_token_id: int = 151652  # Token ID for <|vision_start|>
    vision_end_token_id: int = 151653  # Token ID for <|vision_end|>
