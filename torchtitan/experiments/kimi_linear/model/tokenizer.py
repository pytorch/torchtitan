# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kimi Tokenizer implementation using tiktoken.

This tokenizer is designed to work with Kimi Linear models which use a
tiktoken-based tokenizer with special handling for Chinese characters.
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None
    load_tiktoken_bpe = None


class KimiTokenizer(BaseTokenizer):
    """
    Tokenizer for Kimi Linear models using tiktoken.

    This tokenizer handles:
    - tiktoken BPE encoding
    - Special tokens from tokenizer_config.json
    - Chinese character tokenization with special patterns
    - BOS/EOS token handling

    Args:
        tokenizer_path: Path to directory containing tiktoken.model and config files
    """

    # Kimi-specific tokenization pattern that handles Chinese characters
    pat_str = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    num_reserved_special_tokens = 256

    def __init__(self, tokenizer_path: str):
        super().__init__()

        if not HAS_TIKTOKEN:
            raise ImportError(
                "tiktoken is required for KimiTokenizer. "
                "Please install it: `pip install tiktoken`"
            )

        self.tokenizer_path = tokenizer_path

        # Load configuration
        self.config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )

        # Find and load tiktoken model file
        vocab_file = self._find_vocab_file(tokenizer_path)
        if vocab_file is None:
            raise FileNotFoundError(
                f"Could not find tiktoken.model in {tokenizer_path}"
            )

        # Load mergeable ranks from tiktoken file
        mergeable_ranks = load_tiktoken_bpe(vocab_file)
        num_base_tokens = len(mergeable_ranks)

        # Build special tokens mapping
        self.special_tokens = self._build_special_tokens(num_base_tokens)

        # Create tiktoken encoding
        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words = self.model.n_vocab

        # Extract BOS/EOS/PAD/UNK token IDs
        self._setup_special_token_ids()

        logger.info(
            f"Loaded Kimi tokenizer from {tokenizer_path} - "
            f"vocab_size: {self.n_words}, bos_id: {self.bos_id}, eos_id: {self.eos_id}"
        )

    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from JSON file if it exists."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _find_vocab_file(self, tokenizer_path: str) -> Optional[str]:
        """Find the tiktoken model file in the tokenizer directory."""
        # Try common names
        for name in ["tiktoken.model", "tokenizer.model", "vocab.tiktoken"]:
            path = os.path.join(tokenizer_path, name)
            if os.path.exists(path):
                return path

        # Search for any .model file
        for f in os.listdir(tokenizer_path):
            if f.endswith(".model"):
                return os.path.join(tokenizer_path, f)

        return None

    def _build_special_tokens(self, num_base_tokens: int) -> dict[str, int]:
        """Build special tokens mapping from config."""
        special_tokens = {}

        # Get added_tokens_decoder from config if available
        if self.config and "added_tokens_decoder" in self.config:
            added_tokens_decoder = self.config["added_tokens_decoder"]
            for token_id_str, token_info in added_tokens_decoder.items():
                token_id = int(token_id_str)
                if isinstance(token_info, dict) and "content" in token_info:
                    content = token_info["content"]
                    special_tokens[content] = token_id
        else:
            # Fallback: create default special tokens
            for i in range(
                num_base_tokens, num_base_tokens + self.num_reserved_special_tokens + 2
            ):
                special_tokens[f"<|reserved_token_{i}|>"] = i

        return special_tokens

    def _get_token_from_config(self, key: str) -> Optional[str]:
        """Get a token string from config, handling both string and dict formats."""
        if not self.config:
            return None

        token = self.config.get(key)
        if isinstance(token, dict):
            return token.get("content")
        elif isinstance(token, str):
            return token
        return None

    def _setup_special_token_ids(self):
        """Setup BOS, EOS, PAD, UNK token IDs from config."""
        # Get token strings from config
        bos_token = self._get_token_from_config("bos_token")
        eos_token = self._get_token_from_config("eos_token")
        pad_token = self._get_token_from_config("pad_token")
        unk_token = self._get_token_from_config("unk_token")

        # Map to IDs
        self.bos_id = self.special_tokens.get(bos_token) if bos_token else None
        self.eos_id = self.special_tokens.get(eos_token) if eos_token else None
        self.pad_id = self.special_tokens.get(pad_token) if pad_token else None
        self.unk_id = self.special_tokens.get(unk_token) if unk_token else None

        # Also check config for explicit IDs
        if self.config:
            if self.bos_id is None and "bos_token_id" in self.config:
                self.bos_id = self.config["bos_token_id"]
            if self.eos_id is None and "eos_token_id" in self.config:
                self.eos_id = self.config["eos_token_id"]
            if self.pad_id is None and "pad_token_id" in self.config:
                self.pad_id = self.config["pad_token_id"]

        # Ensure eos_id is set (required by BaseTokenizer)
        if self.eos_id is None:
            # Try to find any EOS-like token
            for token, tid in self.special_tokens.items():
                if "eos" in token.lower() or "end" in token.lower():
                    self.eos_id = tid
                    break

        # Set default if still None
        if self.eos_id is None:
            self.eos_id = 0

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string so that each substring contains no more than
        max_consecutive_slice_len consecutive whitespaces or non-whitespaces.
        """
        if len(s) == 0:
            return

        current_slice_len = 0
        current_slice_is_space = s[0].isspace()
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        **kwargs,
    ) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: The text to encode
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        # Handle large texts by chunking
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        MAX_NO_WHITESPACES_CHARS = 25_000

        all_substrs = []
        for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
            chunk = text[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
            substrs = self._split_whitespaces_or_nonwhitespaces(
                chunk, MAX_NO_WHITESPACES_CHARS
            )
            all_substrs.extend(substrs)

        token_ids: list[int] = []
        for substr in all_substrs:
            token_ids.extend(
                self.model.encode(substr, allowed_special="all")
            )

        # Add BOS/EOS tokens if requested
        if add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: Union[int, list[int]], **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Single token ID or list of token IDs

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        return self.model.decode(token_ids)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.n_words

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.n_words

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        # Check special tokens first
        if token in self.special_tokens:
            return self.special_tokens[token]
        # Encode single token
        encoded = self.model.encode(token, allowed_special="all")
        return encoded[0] if len(encoded) == 1 else None

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert a token ID to its string."""
        try:
            return self.model.decode([token_id])
        except Exception:
            return None


def build_kimi_tokenizer(
    job_config: JobConfig,
) -> Optional[KimiTokenizer]:
    """
    Build a KimiTokenizer from the job config.

    Args:
        job_config: Job configuration containing model.hf_assets_path

    Returns:
        KimiTokenizer instance or None if no path specified
    """
    if not job_config.model.hf_assets_path:
        return None

    tokenizer_path = job_config.model.hf_assets_path

    # Check if this looks like a Kimi tokenizer (has tiktoken.model)
    tiktoken_file = os.path.join(tokenizer_path, "tiktoken.model")
    if os.path.exists(tiktoken_file):
        logger.info(f"Loading Kimi tokenizer from {tokenizer_path}")
        return KimiTokenizer(tokenizer_path)

    # Fall back to trying KimiTokenizer anyway (it will search for model files)
    try:
        return KimiTokenizer(tokenizer_path)
    except FileNotFoundError:
        # If no tiktoken model found, the HuggingFace tokenizer might work
        logger.warning(
            f"No tiktoken model found in {tokenizer_path}, "
            "falling back to HuggingFace tokenizer"
        )
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        return HuggingFaceTokenizer(tokenizer_path)
