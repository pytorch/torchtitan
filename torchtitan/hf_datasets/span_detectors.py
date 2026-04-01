# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torchtitan.tools.logging import logger, warn_once

if TYPE_CHECKING:
    from tokenizers import Encoding

SpanDetectionModelType = Literal[
    "debugmodel",
    "gpt_oss",
    "qwen3",
    "llama3",
    "llama4",
    "deepseek_v3",
]


def _assistant_turns(messages: list[dict[str, str]]) -> int:
    return sum(msg["role"] == "assistant" for msg in messages)


@dataclass(frozen=True, slots=True)
class RegexSpanDetector:
    mode_name: str
    assistant_block_re: re.Pattern[str]
    content_groups: tuple[str, ...] = ("content",)

    def find_char_spans(
        self, messages: list[dict[str, str]], full_text: str
    ) -> list[tuple[int, int]]:
        assistant_blocks = list(self.assistant_block_re.finditer(full_text))
        assistant_turns = _assistant_turns(messages)
        if len(assistant_blocks) != assistant_turns:
            warn_once(
                logger,
                f"{self.mode_name} span detection found a different number of "
                "assistant blocks than assistant turns. Continuing with the "
                "regex-derived spans, which may change masking within affected "
                "examples when prompt or response text reproduces control "
                "sequences.",
            )
        spans = []
        for match in assistant_blocks:
            for group in self.content_groups:
                if match.groupdict().get(group) is None:
                    continue
                spans.append((match.start(group), match.end(group)))
        return spans


# These regexes use lookahead to prefer structural turn endings over literal
# delimiter tokens that may appear inside assistant content. They still cannot
# disambiguate prompt or assistant text that reproduces an entire assistant-turn
# boundary sequence (for example, a user quoting "<|im_start|>assistant" or
# "<|channel|>analysis<|message|>"). That limitation is accepted here as the
# tradeoff for a lightweight, model-specific detector, so collisions degrade
# masking within the affected sample instead of failing preprocessing. This
# also only works with system / user / assistant roles (e.g., no tool calls),
# but that aligns with current dataset support. Extending to support tools
# should be simple.
SPAN_DETECTORS: dict[str, RegexSpanDetector] = {
    "debugmodel": RegexSpanDetector(
        mode_name="debugmodel",
        assistant_block_re=re.compile(
            r"assistant\n(?P<content>.*?)(?=user\n|$)",
            re.DOTALL,
        ),
    ),
    "gpt_oss": RegexSpanDetector(
        mode_name="GPT-OSS",
        assistant_block_re=re.compile(
            r"(?:(?P<analysis_header><\|channel\|>analysis<\|message\|>)"
            r"(?P<analysis_content>.*?<\|end\|>\s*)"
            r"(?=<\|start\|>assistant<\|channel\|>final<\|message\|>))?"
            r"(?P<final_header><\|start\|>assistant<\|channel\|>final<\|message\|>)"
            r"(?P<final_content>.*?<\|(?:end|return)\|>)"
            r"(?=<\|start\|>user<\|message\|>|$)",
            re.DOTALL,
        ),
        content_groups=("analysis_content", "final_content"),
    ),
    "qwen3": RegexSpanDetector(
        mode_name="Qwen3",
        assistant_block_re=re.compile(
            r"<\|im_start\|>assistant\n"
            r"(?P<content>.*?<\|im_end\|>)"
            r"(?=\n<\|im_start\|>user\n|$)",
            re.DOTALL,
        ),
    ),
    "llama3": RegexSpanDetector(
        mode_name="Llama3",
        assistant_block_re=re.compile(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n"
            r"(?P<content>.*?<\|eot_id\|>)"
            r"(?=<\|start_header_id\|>user<\|end_header_id\|>\n\n|$)",
            re.DOTALL,
        ),
    ),
    "llama4": RegexSpanDetector(
        mode_name="Llama4",
        assistant_block_re=re.compile(
            r"<\|header_start\|>assistant<\|header_end\|>\n\n"
            r"(?P<content>.*?<\|eot\|>)"
            r"(?=<\|header_start\|>user<\|header_end\|>\n\n|$)",
            re.DOTALL,
        ),
    ),
    "deepseek_v3": RegexSpanDetector(
        mode_name="DeepSeek V3",
        assistant_block_re=re.compile(
            r"<｜Assistant｜>(?P<content>.*?<｜end▁of▁sentence｜>)(?=<｜User｜>|$)",
            re.DOTALL,
        ),
    ),
}


def char_spans_to_token_spans(
    encoding: "Encoding", char_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Map ordered character spans onto overlapping token spans."""
    offsets = encoding.offsets
    token_spans = []
    token_idx = 0

    for char_start, char_end in char_spans:
        while token_idx < len(offsets) and offsets[token_idx][1] <= char_start:
            token_idx += 1

        span_start = token_idx
        while (
            span_start < len(offsets)
            and offsets[span_start][0] < char_end
            and offsets[span_start][1] <= char_start
        ):
            span_start += 1

        if (
            span_start == len(offsets)
            or offsets[span_start][0] >= char_end
            or offsets[span_start][1] <= char_start
        ):
            raise ValueError(
                f"Could not map chars [{char_start}, {char_end}) to token indices"
            )

        span_end = span_start
        while span_end < len(offsets) and offsets[span_end][0] < char_end:
            if offsets[span_end][1] > char_start:
                span_end += 1
                continue
            break

        token_spans.append((span_start, span_end))
        token_idx = span_end

    return token_spans


def get_span_detector(
    model_type: SpanDetectionModelType | None,
) -> RegexSpanDetector | None:
    if model_type is None:
        return None

    try:
        return SPAN_DETECTORS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported span detection model type '{model_type}'. "
            f"Supported values: {sorted(SPAN_DETECTORS)}"
        ) from exc
