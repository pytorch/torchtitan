# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast, ClassVar

import jinja2
import jinja2.ext
import jinja2.sandbox

from torchtitan.components.tokenizer import MultiModalTokenizer


class _GenerationExtension(jinja2.ext.Extension):
    tags = {"generation"}

    def parse(self, parser: Any) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(
            ["name:endgeneration"],
            drop_needle=True,
        )
        return jinja2.nodes.CallBlock(
            cast(Any, self).call_method("_render_generation"),
            [],
            [],
            body,
        ).set_lineno(lineno)

    def _render_generation(self, caller: Callable[[], str]) -> str:
        return caller()


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


class MuseSparkTokenizer(MultiModalTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalTokenizer.Config):
        audio_token: str
        audio_start_token: str
        audio_end_token: str

    TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        *MultiModalTokenizer.TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )
    LOSS_MASK_TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        *MultiModalTokenizer.LOSS_MASK_TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )

    def set_chat_template(self, template: str) -> None:
        """Compile Muse templates while rendering generation blocks normally."""
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[_GenerationExtension, jinja2.ext.loopcontrols],
        )
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        env.filters["tojson"] = _tojson
        self._chat_template = env.from_string(template)


__all__ = ["MuseSparkTokenizer"]
