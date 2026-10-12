# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer ATEM tool-call parse/render (the novel core of the Muse Glimmer renderer).

Muse Glimmer emits tool calls in Anthropic-style ATEM XML inside the harmony envelope:

    <atem:function_calls>
    <atem:invoke name="search">
    <atem:parameter name="query">who wrote Blade Runner</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

parse: model text -> [{"name", "arguments"}]  (what env.step() needs)
render: a tool call -> ATEM text  (what goes back into the prompt)
"""

from __future__ import annotations

import json
import re

_FUNCTION_CALLS = re.compile(
    r"<atem:function_calls>(.*?)</atem:function_calls>", re.DOTALL
)
_INVOKE = re.compile(
    r'<atem:invoke name="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>', re.DOTALL
)
_PARAMETER = re.compile(
    r'<atem:parameter name="(?P<key>[^"]+)">(?P<value>.*?)</atem:parameter>', re.DOTALL
)


def parse_atem_tool_calls(text: str) -> list[dict]:
    """Parse every ATEM tool call in `text` into `[{"name", "arguments"}]`.

    Values are JSON-decoded when possible (dicts/lists/numbers/bools), else kept
    as the raw string. Supports multiple parallel invokes in one block.
    """
    calls: list[dict] = []
    for block in _FUNCTION_CALLS.findall(text):
        for invoke in _INVOKE.finditer(block):
            arguments: dict = {}
            for param in _PARAMETER.finditer(invoke.group("body")):
                raw = param.group("value")
                try:
                    arguments[param.group("key")] = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    arguments[param.group("key")] = raw
            calls.append({"name": invoke.group("name"), "arguments": arguments})
    return calls


def render_atem_tool_call(name: str, arguments: dict) -> str:
    """Render one tool call as ATEM text (matches Muse Glimmer's chat template)."""
    lines = ["<atem:function_calls>", f'<atem:invoke name="{name}">']
    for key, value in arguments.items():
        if isinstance(value, bool):
            sval = "true" if value else "false"
        elif value is None:
            sval = "null"
        elif isinstance(value, (dict, list)):
            sval = json.dumps(value)
        else:
            sval = str(value)
        lines.append(f'<atem:parameter name="{key}">{sval}</atem:parameter>')
    lines += ["</atem:invoke>", "</atem:function_calls>"]
    return "\n".join(lines)


if __name__ == "__main__":
    # round-trip self-test (no deps, no GPU)
    sample = (
        'thinking...\n<atem:function_calls>\n<atem:invoke name="search">\n'
        '<atem:parameter name="query">who wrote Blade Runner</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    calls = parse_atem_tool_calls(sample)
    assert calls == [
        {"name": "search", "arguments": {"query": "who wrote Blade Runner"}}
    ], calls
    # no tool call -> empty (this is how env.step() detects "final answer")
    assert parse_atem_tool_calls("Philip K. Dick") == []
    # render -> parse round-trip
    rendered = render_atem_tool_call("search", {"query": "x", "topk": 3})
    assert parse_atem_tool_calls(rendered)[0]["arguments"]["topk"] == 3
    print("muse_glimmer atem: all checks passed")
