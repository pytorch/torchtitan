# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from tokenizers import Tokenizer as HfTokenizer
from typing_extensions import override


class Tokenizer(ABC):
    # basic tokenizer interface, for typing purpose mainly
    def __init__(self):
        self._n_words = 8
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @property
    def n_words(self) -> int:
        return self._n_words


class TorchTitanTokenizer(Tokenizer):
    """
    A tokenizer wrapper that handles BOS/EOS token inference and encoding.

    This class loads tokenizer files and automatically infers BOS/EOS tokens from
    configuration file. It provides an encode method that adds
    BOS/EOS tokens based on whether the underlying tokenizer adds them automatically.

    Args:
        tokenizer_path (str): Path to directory containing tokenizer files
    """

    def __init__(
        self,
        tokenizer_path: str,
    ):
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

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> HfTokenizer:
        """Load tokenizer from various file formats."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        # Define paths for different tokenizer file types
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer_model_path = os.path.join(tokenizer_path, "tokenizer.model")
        vocab_txt_path = os.path.join(tokenizer_path, "vocab.txt")
        vocab_json_path = os.path.join(tokenizer_path, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_path, "merges.txt")

        try:
            # Strategy 1: Load from tokenizer.json (preferred for modern tokenizers)
            if os.path.exists(tokenizer_json_path):
                print("Loading tokenizer from tokenizer.json")
                return HfTokenizer.from_file(tokenizer_json_path)

            # Strategy 2: Load from tokenizer.model (SentencePiece models like Llama)
            elif os.path.exists(tokenizer_model_path):
                print("Loading tokenizer from tokenizer.model")
                return HfTokenizer.from_file(tokenizer_model_path)

            # Strategy 3: Load from vocab files (with or without merges.txt)
            elif os.path.exists(vocab_json_path) or os.path.exists(vocab_txt_path):
                from tokenizers import Tokenizer

                # Load vocabulary
                if os.path.exists(vocab_json_path):
                    print("Loading vocabulary from vocab.json")
                    with open(vocab_json_path, "r") as f:
                        vocab = json.load(f)
                    vocab_source = "vocab.json"
                else:
                    print("Loading vocabulary from vocab.txt")
                    vocab = {}
                    with open(vocab_txt_path, "r") as f:
                        for i, line in enumerate(f):
                            token = line.strip()
                            if token:
                                vocab[token] = i
                    vocab_source = "vocab.txt"

                # Strategy 3a: Use BPE if merges.txt exists
                if os.path.exists(merges_txt_path):
                    print(f"Loading BPE tokenizer from {vocab_source} + merges.txt")
                    from tokenizers.models import BPE

                    bpe_model = BPE(vocab=vocab, merges=merges_txt_path)
                    return Tokenizer(bpe_model)

                # Strategy 3b: Use WordLevel if no merges.txt
                else:
                    print(f"Loading WordLevel tokenizer from {vocab_source}")
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
                    "Looking for: tokenizer.json, tokenizer.model, vocab.txt+merges.txt, or vocab.json+merges.txt"
                )

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise e
            raise Exception(
                f"Failed to load tokenizer from '{tokenizer_path}': {e}"
            ) from e

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

    def _infer_special_tokens(self):
        """
        Read special tokens from config and add them to the underlying tokenizer.
        Store BOS/EOS tokens as class attributes since they are frequently used.
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

        # Try to get tokens from tokenizer config
        if self.config:
            for key in standard_keys:
                token_config = self.config.get(key)
                if token_config is not None:
                    # Extract token string
                    token_str = self._get_token_from_config(self.config, key)
                    if token_str is not None:
                        # Store BOS/EOS tokens as class attributes (frequently used)
                        if key == "bos_token":
                            self.bos_token = token_str
                            self.bos_id = self.tokenizer.token_to_id(token_str)
                        elif key == "eos_token":
                            self.eos_token = token_str
                            self.eos_id = self.tokenizer.token_to_id(token_str)

                        # Create AddedToken object with proper configuration
                        if (
                            isinstance(token_config, dict)
                            and token_config.get("__type") == "AddedToken"
                        ):
                            # Extract AddedToken properties from config
                            from tokenizers import AddedToken

                            added_token = AddedToken(
                                content=token_str,
                                single_word=token_config.get("single_word", False),
                                lstrip=token_config.get("lstrip", False),
                                rstrip=token_config.get("rstrip", False),
                                normalized=token_config.get("normalized", True),
                                special=True,  # Mark as special token
                            )
                            added_tokens_to_add.append(added_token)
                        else:
                            # Simple string token - create basic AddedToken
                            from tokenizers import AddedToken

                            added_token = AddedToken(content=token_str, special=True)
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
        self.should_add_bos = False
        self.should_add_eos = False
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
                self.should_add_bos = bool(config_add_bos)
            if config_add_eos is not None:
                self.should_add_eos = bool(config_add_eos)

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

        add_bos = kwargs.get("add_bos", False)
        add_eos = kwargs.get("add_eos", False)

        # Get base token IDs from the underlying tokenizer
        token_ids = self.tokenizer.encode(text).ids

        # Add BOS token if requested and not already added by tokenizer
        if not self.hf_adds_bos and (add_bos or self.should_add_bos):
            if self.bos_id is not None:
                token_ids.insert(0, self.bos_id)

        # Add EOS token if requested and not already added by tokenizer
        if not self.hf_adds_eos and (add_eos or self.should_add_eos):
            if self.eos_id is not None:
                token_ids.append(self.eos_id)

        return token_ids

    @override
    def decode(self, *args, **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids (list[int]): List of token IDs to decode

        Returns:
            str: Decoded text
        """
        # Extract token_ids from arguments
        if len(args) >= 1:
            token_ids = args[0]
        else:
            token_ids = kwargs.get("token_ids", [])

        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        return self.tokenizer.id_to_token(token_id)


def load_tokenizer_config(tokenizer_path: str) -> dict:
    """Load tokenizer configuration from tokenizer_config.json if it exists."""
    config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def load_tokenizer(tokenizer_path: str) -> TorchTitanTokenizer:
    """
    Load a TorchTitanTokenizer from the specified path.

    This function creates a TorchTitanTokenizer instance that handles BOS/EOS token
    inference and intelligent encoding. The tokenizer automatically detects and loads
    from various file formats and infers special token behavior.

    Args:
        tokenizer_path (str): Path to the directory containing tokenizer files.
                             Should contain one or more of the supported file types.

    Returns:
        tokenizer (TorchTitanTokenizer): Loaded tokenizer instance with intelligent BOS/EOS handling
    """
    tokenizer = TorchTitanTokenizer(tokenizer_path)
    return tokenizer
