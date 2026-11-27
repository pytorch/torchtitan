# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer

_GENERATION_TAG = re.compile(r"(\{%[-+]?\s*)(end)?generation(\s*[-+]?%\})")


class InklingTokenizer(HuggingFaceTokenizer):
    """Load the published tokenizer without requiring the Transformers renderer."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def set_chat_template(self, template: str) -> None:
        # HF generation tags track assistant spans. Plain rendering preserves
        # their contents and whitespace but does not need to record the spans.
        template = _GENERATION_TAG.sub(
            lambda tag: tag[1] + ("endwith" if tag[2] else "with") + tag[3], template
        )
        super().set_chat_template(template)
