# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer
from torchtitan_recipes.models.muse_spark import MUSE_SPARK_SPECIAL_TOKENS
from torchtitan_recipes.tests.models.muse_spark import MUSE_SPARK_DEBUG_SPECIAL_TOKENS


@pytest.fixture
def tokenizer():
    return MuseSparkTokenizer.Config(**MUSE_SPARK_DEBUG_SPECIAL_TOKENS).build(
        tokenizer_path="tests/assets/tokenizer"
    )


def test_multimodal_tokens_and_loss_mask(tokenizer):
    for field, token in MUSE_SPARK_DEBUG_SPECIAL_TOKENS.items():
        prefix = field.removesuffix("_token")
        token_id = tokenizer.token_to_id(token)
        assert getattr(tokenizer, f"{prefix}_token") == token
        assert getattr(tokenizer, f"{prefix}_id") == token_id
        assert tokenizer.encode(token, add_bos=False, add_eos=False) == [token_id]
    assert {"audio", "audio_start", "audio_end"} <= set(
        tokenizer.LOSS_MASK_TOKEN_FIELDS
    )
    assert "pad" not in tokenizer.LOSS_MASK_TOKEN_FIELDS


@pytest.mark.parametrize("field", MUSE_SPARK_SPECIAL_TOKENS)
def test_missing_special_token_is_rejected(field):
    tokens = MUSE_SPARK_DEBUG_SPECIAL_TOKENS | {field: MUSE_SPARK_SPECIAL_TOKENS[field]}
    with pytest.raises(ValueError, match=f"config field '{field}'"):
        MuseSparkTokenizer.Config(**tokens).build(
            tokenizer_path="tests/assets/tokenizer"
        )


def test_generation_blocks_and_json_filter(tokenizer):
    tokenizer.set_chat_template(
        "{% for message in messages %}{{ message.role }}:"
        "{% generation %}{{ message.content | tojson }}{% endgeneration %}"
        "{% endfor %}"
    )
    content = {"query": 'a "quoted" value', "items": [1, 2]}
    rendered = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": content}]
    )
    assert rendered == "assistant:" + json.dumps(content)
