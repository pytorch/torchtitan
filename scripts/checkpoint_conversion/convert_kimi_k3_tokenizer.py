#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a Hugging Face ``tokenizer.json`` for Kimi K3 and check it and the renderer.

Kimi K3 ships a tiktoken BPE (``tiktoken.model``) with remote tokenizer code, but
TorchTitan's ``HuggingFaceTokenizer`` loads ``tokenizer.json``. This script converts
the BPE with transformers' ``TikTokenConverter``, adds the 256 special tokens at the
ids K3 assigns them, and then checks against the checkpoint's own code:

- the converted tokenizer gives the same ids as K3's tokenizer on mixed texts,
  including the XTML chat markers;
- ``KimiK3Renderer`` gives the same token ids as the checkpoint's
  ``apply_chat_template`` for single-turn, system and multi-turn conversations, with
  and without thinking.

Install the reference dependencies with::

    pip install tiktoken==0.14.0 transformers==4.56.2

Usage:
    python -m scripts.checkpoint_conversion.convert_kimi_k3_tokenizer \
        <Kimi-K3 checkpoint dir> <output dir>

The output directory receives ``tokenizer.json``. Point TorchTitan's
``hf_assets_path`` at a directory holding it next to the checkpoint's config and
weights (symlinks are enough).
"""

import argparse
from pathlib import Path

from tokenizers import AddedToken
from torchtitan.components.renderer import RendererTokenizerWrapper
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.kimi_k3.renderer import KimiK3Renderer, KimiK3RendererConfig
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import TikTokenConverter

_TEXTS = [
    "Hello, world! What's 2+2? It's 4.",
    "Let $f(x) = x^2 + 3x - 7$. Find all real $x$ such that $f(x) = 11$. \\boxed{2}",
    "    def foo(bar):\n        return bar ** 2  # comment\n\n\n",
    "数学题：求解方程 x^2 - 5x + 6 = 0 的根。",
    "Mixed 中文 and English, 12345678 numbers, émojis \U0001f642 and tabs\t\tend",
    "CamelCaseWords and ALLCAPS and don't won't they'll I'm we've",
    "\r\n\r\n  trailing spaces   \n",
    "<|open|>user<|sep|>Solve 1+1.<|close|><|end_of_msg|>",
    "[BOS]text[EOS]",
]

_QUESTION = (
    "Let $x$ and $y$ be real numbers such that $x + y = 10$ and $xy = 21$. "
    "Find $x^2 + y^2$. Put your final answer within \\boxed{}."
)
_CONVERSATIONS = [
    [{"role": "user", "content": _QUESTION}],
    [
        {"role": "system", "content": "You are a careful mathematician."},
        {"role": "user", "content": _QUESTION},
    ],
    [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "One plus one."},
        {"role": "user", "content": "And 2+2?"},
    ],
]


def convert(checkpoint: Path, output: Path, reference) -> None:
    """Write ``output / tokenizer.json`` and check its ids against ``reference``."""
    special_tokens = sorted(reference.special_tokens.items(), key=lambda kv: kv[1])
    num_base_tokens = special_tokens[0][1]
    if [token_id for _, token_id in special_tokens] != list(
        range(num_base_tokens, num_base_tokens + len(special_tokens))
    ):
        raise ValueError("special tokens must follow the base vocabulary contiguously")
    tokenizer = TikTokenConverter(
        vocab_file=str(checkpoint / "tiktoken.model"),
        pattern=type(reference).pat_str,
    ).converted()
    # Added tokens take ids from the end of the base vocabulary, as K3 assigns them.
    if tokenizer.get_vocab_size(with_added_tokens=True) != num_base_tokens:
        raise ValueError("unexpected base vocabulary size")
    tokenizer.add_special_tokens(
        [
            AddedToken(token, special=True, normalized=False)
            for token, _ in special_tokens
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"))

    for token, token_id in special_tokens:
        if tokenizer.token_to_id(token) != token_id:
            raise ValueError(f"{token} maps to {tokenizer.token_to_id(token)}")
    mismatches = [
        text
        for text in _TEXTS
        if reference.encode(text, add_special_tokens=False)
        != tokenizer.encode(text, add_special_tokens=False).ids
    ]
    if mismatches:
        raise ValueError(f"ids differ from K3's tokenizer on {mismatches}")
    print(
        f"tokenizer.json: {tokenizer.get_vocab_size()} tokens "
        f"({num_base_tokens} base + {len(special_tokens)} special); "
        f"{len(_TEXTS)} texts match K3's tokenizer"
    )


def check_renderer(output: Path, reference) -> None:
    """Check ``KimiK3Renderer`` against the checkpoint's ``apply_chat_template``."""
    tokenizer = RendererTokenizerWrapper(
        HuggingFaceTokenizer(tokenizer_path=str(output))
    )
    for thinking, effort in ((True, "max"), (True, "low"), (False, "max")):
        renderer = KimiK3Renderer(
            tokenizer, KimiK3RendererConfig(thinking=thinking, thinking_effort=effort)
        )
        for conversation in _CONVERSATIONS:
            expected = reference.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                thinking=thinking,
                thinking_effort=effort,
            )
            if not isinstance(expected, list):
                expected = expected["input_ids"]
            rendered = renderer.render_ids(conversation, add_generation_prompt=True)
            if rendered != expected:
                raise ValueError(
                    f"renderer differs from apply_chat_template (thinking={thinking}, "
                    f"effort={effort}, {len(conversation)} messages)"
                )
    print(
        f"KimiK3Renderer matches apply_chat_template on "
        f"{3 * len(_CONVERSATIONS)} conversation settings"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="Kimi K3 checkpoint directory")
    parser.add_argument("output", type=Path, help="Directory for tokenizer.json")
    args = parser.parse_args()
    reference = AutoTokenizer.from_pretrained(
        str(args.checkpoint), trust_remote_code=True
    )
    convert(args.checkpoint, args.output, reference)
    check_renderer(args.output, reference)


if __name__ == "__main__":
    main()
