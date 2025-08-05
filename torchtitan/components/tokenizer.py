# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from tokenizers import AddedToken, Tokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger
from typing_extensions import override


class BaseTokenizer(ABC):
    # base tokenizer interface, for typing purpose mainly
    def __init__(self):
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...


class HuggingFaceTokenizer(BaseTokenizer):
    """
    A tokenizer wrapper that handles BOS/EOS token inference and encoding.

    This class loads tokenizer files and automatically infers BOS/EOS tokens from
    a configuration file (tokenizer_config.json). It provides an encode method that adds
    BOS/EOS tokens based on whether the underlying tokenizer adds them automatically.

    Args:
        tokenizer_path (str): Path to directory containing tokenizer files
    """

    def __init__(
        self,
        tokenizer_path: str,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path

        # Initialize BOS/EOS token attributes (frequently used)
        self.bos_id = None
        self.eos_id = None
        self.bos_token = None
        self.eos_token = None

        # Load the underlying tokenizer
        self.tokenizer = self._load_tokenizer_from_path(tokenizer_path)

        # Load configuration files
        self.config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )

        # Infer special tokens and adding BOS/EOS behavior
        self._infer_special_tokens()
        self._infer_should_add_bos_eos()

    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from JSON file if it exists."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> Tokenizer:
        """Load tokenizer from various file formats."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        # Define paths for different tokenizer file types
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        vocab_txt_path = os.path.join(tokenizer_path, "vocab.txt")
        vocab_json_path = os.path.join(tokenizer_path, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_path, "merges.txt")

        # Strategy 1: Load from tokenizer.json (preferred for modern tokenizers)
        if os.path.exists(tokenizer_json_path):
            logger.info("Loading tokenizer from tokenizer.json")
            return Tokenizer.from_file(tokenizer_json_path)
        # Strategy 2: Load from vocab files (with or without merges.txt)
        elif os.path.exists(vocab_json_path) or os.path.exists(vocab_txt_path):
            # Load vocabulary
            if os.path.exists(vocab_json_path):
                logger.info("Loading vocabulary from vocab.json")
                with open(vocab_json_path, "r") as f:
                    vocab = json.load(f)
                vocab_source = "vocab.json"
            else:
                logger.info("Loading vocabulary from vocab.txt")
                vocab = {}
                with open(vocab_txt_path, "r") as f:
                    for i, line in enumerate(f):
                        token = line.strip()
                        if token:
                            vocab[token] = i
                vocab_source = "vocab.txt"

            # Strategy 2a: Use BPE if merges.txt exists
            if os.path.exists(merges_txt_path):
                logger.info(f"Loading BPE tokenizer from {vocab_source} + merges.txt")
                from tokenizers import decoders, pre_tokenizers, processors
                from tokenizers.models import BPE

                # Load merges from file and convert to tuples
                merges = []
                with open(merges_txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(
                            "#"
                        ):  # Skip comments and empty lines
                            parts = line.split()
                            if len(parts) >= 2:
                                merges.append((parts[0], parts[1]))

                # Create BPE model
                bpe_model = BPE(vocab=vocab, merges=merges)
                tokenizer = Tokenizer(bpe_model)

                # Configure GPT-2 style components for proper space handling
                tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False
                )
                tokenizer.decoder = decoders.ByteLevel()
                tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

                return tokenizer

            # Strategy 2b: Use WordLevel if no merges.txt
            else:
                logger.info(f"Loading WordLevel tokenizer from {vocab_source}")
                from tokenizers.models import WordLevel

                word_level_model = WordLevel(vocab=vocab, unk_token="[UNK]")
                return Tokenizer(word_level_model)

        else:
            # List available files for debugging
            available_files = [
                f
                for f in os.listdir(tokenizer_path)
                if os.path.isfile(os.path.join(tokenizer_path, f))
            ]
            raise FileNotFoundError(
                f"No supported tokenizer files found in '{tokenizer_path}'. "
                f"Available files: {available_files}. "
                "Looking for: tokenizer.json, vocab.txt+merges.txt, or vocab.json+merges.txt"
            )

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> Optional[str]:
        """
        Parse special tokens from config that can be either strings or dicts.
        HF tokens are stored as either {'bos_token': '<bos>'} or {'bos_token': {'content': '<bos>', ...}}.
        """
        token = config.get(key)
        if isinstance(token, dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
        elif token is not None and not isinstance(token, str):
            raise ValueError(
                f"Could not parse {key} from config - expected string or dict"
            )
        return token

    def _process_special_token(
        self, token_str: str, token_config: dict, token_id: Optional[int] = None
    ) -> AddedToken:
        """
        Process a special token and update BOS/EOS attributes if applicable.

        Args:
            token_str: The token string content
            token_config: Token configuration dictionary
            token_id: Optional explicit token ID (for added_tokens_decoder)

        Returns:
            AddedToken object to be added to the tokenizer
        """
        # Get reference BOS/EOS tokens from config for comparison
        config_bos_token = (
            self._get_token_from_config(self.config, "bos_token")
            if self.config
            else None
        )
        config_eos_token = (
            self._get_token_from_config(self.config, "eos_token")
            if self.config
            else None
        )

        # Store BOS/EOS tokens as class attributes if they match
        if token_str == config_bos_token:
            self.bos_token = token_str
            self.bos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )
        elif token_str == config_eos_token:
            self.eos_token = token_str
            self.eos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )

        # Create AddedToken object based on config format
        if isinstance(token_config, dict):
            if token_config.get("__type") == "AddedToken" or "content" in token_config:
                # Handle both AddedToken format and added_tokens_decoder format
                return AddedToken(
                    content=token_str,
                    single_word=token_config.get("single_word", False),
                    lstrip=token_config.get("lstrip", False),
                    rstrip=token_config.get("rstrip", False),
                    normalized=token_config.get("normalized", True),
                    special=token_config.get("special", True),
                )

        # Fallback to simple special token
        return AddedToken(content=token_str, special=True)

    def _infer_special_tokens(self):
        """
        Read special tokens from config and add them to the underlying tokenizer.
        Store BOS/EOS tokens as class attributes since they are frequently used.

        This method handles multiple token configuration formats:
        1. Standard top-level keys (bos_token, eos_token, etc.)
        2. added_tokens_decoder dictionary (used by models like Llama 3.1)
        """
        standard_keys = [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]

        # List to collect AddedToken objects for updating the underlying tokenizer
        added_tokens_to_add = []

        if not self.config:
            return

        # Process standard top-level token keys
        for key in standard_keys:
            token_config = self.config.get(key)
            if token_config is not None:
                token_str = self._get_token_from_config(self.config, key)
                if token_str is not None:
                    added_token = self._process_special_token(token_str, token_config)
                    added_tokens_to_add.append(added_token)

        # Process added_tokens_decoder (comprehensive special token definitions)
        added_tokens_decoder = self.config.get("added_tokens_decoder", {})
        for token_id_str, token_config in added_tokens_decoder.items():
            if isinstance(token_config, dict) and "content" in token_config:
                token_str = token_config["content"]
                token_id = int(token_id_str)
                added_token = self._process_special_token(
                    token_str, token_config, token_id
                )
                added_tokens_to_add.append(added_token)

        # Update the underlying tokenizer with special tokens
        if added_tokens_to_add:
            self.tokenizer.add_special_tokens(added_tokens_to_add)

            # Update BOS/EOS token IDs after adding to tokenizer (in case they changed)
            if self.bos_token:
                self.bos_id = self.tokenizer.token_to_id(self.bos_token)
            if self.eos_token:
                self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    def _infer_should_add_bos_eos(self):
        """
        Determine if we should add BOS/EOS tokens based on config settings.
        If config explicitly specifies add_bos_token/add_eos_token, follow that.
        Otherwise, determine if the underlying tokenizer automatically adds them.
        """
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        # First, determine if underlying tokenizer auto-adds BOS/EOS tokens empirically
        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        # Check tokenizer_config.json for explicit settings - these override empirical detection
        if self.config:
            config_add_bos = self.config.get("add_bos_token")
            config_add_eos = self.config.get("add_eos_token")
            if config_add_bos is not None:
                self.default_add_bos = bool(config_add_bos)
            if config_add_eos is not None:
                self.default_add_eos = bool(config_add_eos)

    def encode(self, *args, **kwargs) -> list[int]:
        """
        Encode text into token IDs with BOS/EOS handling.

        Args:
            text (str): The text to encode
            add_bos (bool): Whether to add BOS token (if not already added by tokenizer)
            add_eos (bool): Whether to add EOS token (if not already added by tokenizer)

        Returns:
            list[int]: List of token IDs
        """
        # Extract arguments
        if len(args) >= 1:
            text = args[0]
        else:
            text = kwargs.get("text", "")

        add_bos = kwargs.get("add_bos", self.default_add_bos)
        add_eos = kwargs.get("add_eos", self.default_add_eos)

        # Get base token IDs from the underlying tokenizer
        token_ids = self.tokenizer.encode(text).ids

        # Add BOS token if requested and not already added by tokenizer
        if not self.hf_adds_bos and add_bos:
            if self.bos_id is not None:
                token_ids.insert(0, self.bos_id)

        # Add EOS token if requested and not already added by tokenizer
        if not self.hf_adds_eos and add_eos:
            if self.eos_id is not None:
                token_ids.append(self.eos_id)

        return token_ids

    @override
    def decode(self, *args, **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids (list[int]): List of token IDs to decode
            **kwargs: Additional arguments passed to the underlying tokenizer's decode method
                     (e.g., skip_special_tokens)

        Returns:
            str: Decoded text
        """
        # Extract token_ids from arguments
        if len(args) >= 1:
            token_ids = args[0]
            # Pass through remaining kwargs
            return self.tokenizer.decode(token_ids, **kwargs)
        else:
            token_ids = kwargs.pop("token_ids", [])
            # Pass through remaining kwargs after removing token_ids
            return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        return self.tokenizer.id_to_token(token_id)


def build_hf_tokenizer(
    job_config: JobConfig,
) -> Union[HuggingFaceTokenizer, BaseTokenizer]:
    """
    Builds a HuggingFaceTokenizer from the specified path.

    This function creates a HuggingFaceTokenizer instance that handles BOS/EOS token
    inference and intelligent encoding. The tokenizer automatically detects and loads
    from various file formats and infers special token behavior.

    Args:
        JobConfig: A JobConfig object containing the path to the tokenizer directory.

    Returns:
        tokenizer (HuggingFaceTokenizer): Loaded tokenizer instance with intelligent BOS/EOS handling
    """
    tokenizer = HuggingFaceTokenizer(job_config.model.tokenizer_path)
    return tokenizer
