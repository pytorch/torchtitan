import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from tokenizers import AddedToken, Tokenizer

from src.logging import logger


def resolve_tokenizer_path(tokenizer: str) -> str:
    """Resolve a tokenizer path: if local directory with tokenizer.json exists,
    use it directly. Otherwise treat as a HF model ID and download."""
    if os.path.isdir(tokenizer) and os.path.exists(
        os.path.join(tokenizer, "tokenizer.json")
    ):
        return tokenizer

    # Treat as HF model ID, download to .tokenizers/<model_id>
    cache_dir = os.path.join(".tokenizers", tokenizer.replace("/", "--"))
    if os.path.exists(os.path.join(cache_dir, "tokenizer.json")):
        logger.info(f"Using cached tokenizer at {cache_dir}")
        return cache_dir

    logger.info(f"Downloading tokenizer from HuggingFace: {tokenizer}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        tokenizer,
        allow_patterns=["tokenizer*", "special_tokens*"],
        local_dir=cache_dir,
    )
    return cache_dir


class BaseTokenizer(ABC):
    """Base tokenizer interface."""

    def __init__(self):
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]: ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def get_vocab_size(self) -> int: ...


class HuggingFaceTokenizer(BaseTokenizer):
    """Tokenizer that loads from a HuggingFace tokenizer.json file.

    Automatically infers BOS/EOS tokens from tokenizer_config.json and handles
    whether the underlying tokenizer adds them automatically.

    Args:
        tokenizer_path: Path to directory containing tokenizer.json
    """

    def __init__(self, tokenizer_path: str):
        super().__init__()
        self.tokenizer_path = tokenizer_path

        # ? we start with none and try to infer they correctly
        self.bos_id = None  # ? begining of sentence token id, a number
        # pyrefly: ignore [bad-assignment]
        self.eos_id = None  # ? end of sentence token id, a number
        self.bos_token = None  # ? begining of sentence token, the actual token
        self.eos_token = None  # ? end of sentence token, the actual token

        # Load tokenizer.json
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"tokenizer.json not found in '{tokenizer_path}'")
        logger.info("Loading tokenizer from tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)

        # Load config and infer special tokens
        config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        self.config = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

        self._infer_special_tokens()
        self._infer_should_add_bos_eos()

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> Optional[str]:
        """Parse special tokens that can be either strings or dicts."""
        token = config.get(key)
        if isinstance(token, dict):
            return token.get("content")
        return token

    def _infer_special_tokens(self):
        """Read special tokens from config and add them to the tokenizer."""
        if not self.config:
            return

        config_bos = self._get_token_from_config(
            self.config, "bos_token"
        )  # ? get the bos token from the config
        config_eos = self._get_token_from_config(
            self.config, "eos_token"
        )  # ? get the eos token from the config
        added_tokens: list[AddedToken] = []

        # Process standard token keys
        # ? walk through possible special tokens
        for key in [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]:
            token_config = self.config.get(key)
            if token_config is None:
                continue
            token_str = self._get_token_from_config(self.config, key)
            if token_str is None:
                continue

            if token_str == config_bos:
                self.bos_token = token_str
                self.bos_id = self.tokenizer.token_to_id(token_str)
            elif token_str == config_eos:
                self.eos_token = token_str
                self.eos_id = self.tokenizer.token_to_id(token_str)

            added_tokens.append(AddedToken(content=token_str, special=True))

        # Process added_tokens_decoder (used by Llama 3.1 etc.)
        for token_id_str, token_config in self.config.get(
            "added_tokens_decoder", {}
        ).items():
            if isinstance(token_config, dict) and "content" in token_config:
                token_str = token_config["content"]
                token_id = int(token_id_str)

                if token_str == config_bos:
                    self.bos_token = token_str
                    self.bos_id = token_id
                elif token_str == config_eos:
                    self.eos_token = token_str
                    self.eos_id = token_id

                added_tokens.append(
                    AddedToken(
                        content=token_str,
                        special=token_config.get("special", True),
                    )
                )

        if added_tokens:
            self.tokenizer.add_special_tokens(added_tokens)
            # Re-resolve IDs after adding
            if self.bos_token:
                self.bos_id = self.tokenizer.token_to_id(self.bos_token)
            if self.eos_token:
                self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    def _infer_should_add_bos_eos(self):
        """Determine if we should manually add BOS/EOS tokens."""
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        # Check if underlying tokenizer auto-adds BOS/EOS
        encoded_empty = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty:
            self.hf_adds_eos = True

        # Config overrides
        if self.config:
            add_bos = self.config.get("add_bos_token")
            add_eos = self.config.get("add_eos_token")
            if add_bos is not None:
                self.default_add_bos = bool(add_bos)
            if add_eos is not None:
                self.default_add_eos = bool(add_eos)

    def encode(self, *args, **kwargs) -> list[int]:
        text = args[0] if args else kwargs.get("text", "")
        add_bos = kwargs.get("add_bos", self.default_add_bos)
        add_eos = kwargs.get("add_eos", self.default_add_eos)

        token_ids = self.tokenizer.encode(text).ids

        if not self.hf_adds_bos and add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)
        if not self.hf_adds_eos and add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, *args, **kwargs) -> str:
        token_ids = args[0] if args else kwargs.pop("token_ids", [])
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
