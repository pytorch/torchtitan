# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rebuild a ``tokenizer.json`` from a Kimi tiktoken vocabulary.

The Kimi family -- Kimi-K2.5, Kimi-VL-A3B and the Moonlight-16B-A3B text
sibling -- ships its vocabulary as ``tiktoken.model``: base64 token bytes plus
a merge rank per line.
``torchtitan.components.tokenizer`` reads ``tokenizer.json`` or a vocab/merges
pair, so those model configs fail while building their tokenizer.

Converting keeps every dependency out of both the runtime and the install: only
``tokenizers``, already required, is used here.
"""

import base64
import json
import os

# The Kimi family's pre-tokenizer regex, from ``pat_str`` in the
# ``tokenization_moonshot`` module its repositories ship. A tiktoken file stores
# no pre-tokenizer, and splitting text differently silently changes
# tokenization, so it must be supplied.
KIMI_PATTERN = "|".join(
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


def _byte_encoder() -> dict[int, str]:
    """Map each byte to a distinct printable character, as GPT-2 BPE does.

    ``tokenizers`` keys its vocabulary by ``str`` while tiktoken tokens are raw
    bytes, so the bytes take this reversible detour through characters.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    codes = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + spare)
            spare += 1
    return {byte: chr(code) for byte, code in zip(printable, codes)}


def _merge_pair(ranks: dict[bytes, int], token: bytes) -> tuple[bytes, bytes]:
    """The two tokens ``token`` was merged from.

    tiktoken records the rank a token was learned at, not the pair that formed
    it. Replaying the merge loop over the token's own bytes and stopping below
    its own rank leaves exactly those two halves.
    """
    parts = [token[i : i + 1] for i in range(len(token))]
    while len(parts) > 2:
        pairs = [
            (ranks[merged], index)
            for index in range(len(parts) - 1)
            if (merged := parts[index] + parts[index + 1]) in ranks
            and ranks[merged] < ranks[token]
        ]
        if not pairs:
            break
        index = min(pairs)[1]
        parts[index : index + 2] = [parts[index] + parts[index + 1]]
    if len(parts) != 2:
        raise ValueError(f"token {token!r} is not a merge of two known tokens")
    return parts[0], parts[1]


def _special_tokens(tokenizer_config_path: str, first_id: int) -> list[str]:
    """Special-token strings for every id from ``first_id`` up, in id order.

    ``added_tokens_decoder`` names only some reserved slots. The unnamed ones
    still occupy ids, so they get placeholders; otherwise every later special
    token would shift down and stop matching the model.
    """
    if not os.path.exists(tokenizer_config_path):
        return []
    with open(tokenizer_config_path) as handle:
        entries = json.load(handle).get("added_tokens_decoder", {})
    named = {
        int(i): entry["content"] for i, entry in entries.items() if int(i) >= first_id
    }
    return [
        named.get(i, f"<|reserved_token_{i}|>")
        for i in range(first_id, max(named, default=first_id - 1) + 1)
    ]


def convert_tiktoken_to_tokenizer_json(
    model_dir: str, *, pattern: str = KIMI_PATTERN
) -> int:
    """Write ``tokenizer.json`` beside the ``tiktoken.model`` in ``model_dir``.

    Args:
        model_dir: Directory holding ``tiktoken.model``, and optionally
            ``tokenizer_config.json`` for special-token ids.
        pattern: Pre-tokenizer regex the model was trained with.

    Returns:
        Total vocabulary size, base plus special tokens.
    """
    from tokenizers import decoders, models, pre_tokenizers, Regex, Tokenizer

    with open(os.path.join(model_dir, "tiktoken.model")) as handle:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in handle if line.strip())
        }
    encoder = _byte_encoder()

    def as_text(token: bytes) -> str:
        return "".join(encoder[byte] for byte in token)

    multi_byte = sorted((t for t in ranks if len(t) > 1), key=ranks.__getitem__)
    tokenizer = Tokenizer(
        models.BPE(
            vocab={as_text(token): rank for token, rank in ranks.items()},
            merges=[
                tuple(as_text(part) for part in _merge_pair(ranks, token))
                for token in multi_byte
            ],
        )
    )
    # Split on the model's pattern first; use_regex=False stops ByteLevel from
    # re-splitting with GPT-2's pattern on top of it.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        _special_tokens(
            os.path.join(model_dir, "tokenizer_config.json"), first_id=len(ranks)
        )
    )

    tokenizer.save(os.path.join(model_dir, "tokenizer.json"))
    return tokenizer.get_vocab_size()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory holding tiktoken.model")
    args = parser.parse_args()
    print(
        f"Wrote tokenizer.json ({convert_tiktoken_to_tokenizer_json(args.model_dir)})"
    )
