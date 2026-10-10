# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


class HFBackendTokenizer(HuggingFaceTokenizer):
    """Tokenizer that passes special tokens to chat templates.

    Some HF models' chat templates reference bos_token, eos_token, etc.
    The base tokenizer only passes messages. This subclass auto-injects
    special tokens so all HF chat templates work.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def __init__(self, config=None, *, tokenizer_path):
        super().__init__(config=config, tokenizer_path=tokenizer_path)
        # Fix: when bos_token and eos_token are the same string, the base
        # tokenizer's elif logic only sets bos_token. Ensure eos is set too.
        if self.eos_id is None and self._hf_config:
            eos_str = self._get_token_from_config(self._hf_config, "eos_token")
            if eos_str is not None:
                self.eos_token = eos_str
                self.eos_id = self.tokenizer.token_to_id(eos_str)

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("bos_token", self.bos_token or "")
        kwargs.setdefault("eos_token", self.eos_token or "")
        return super().apply_chat_template(messages, **kwargs)

    def encode(self, text: str, **kwargs) -> list[int]:
        # Chat templates like Llama 3's render bos_token; don't prepend a second BOS.
        if self.bos_token and text.startswith(self.bos_token):
            kwargs["add_bos"] = False
        return super().encode(text, **kwargs)
