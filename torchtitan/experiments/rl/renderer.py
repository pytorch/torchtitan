# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``RendererConfig`` — message ↔ token conversion via the ``renderers`` library.

The rollout driver calls ``renderer.render_ids`` and
``renderer.parse_response`` from N concurrent rollouts in one event
loop. We use :class:`renderers.RendererPool` (pool size = max
concurrent rollouts) so each thread checks out its own tokenizer slot;
HF fast tokenizers release the GIL, so threads achieve real CPU
parallelism rather than just event-loop responsiveness.

Auto-detection: ``renderer="auto"`` matches the tokenizer's
``name_or_path`` against ``MODEL_RENDERER_MAP`` (Qwen3, GPT-OSS,
Llama-3, Kimi, GLM, etc.). Falls back to ``DefaultRenderer``
(``apply_chat_template`` verbatim) for unknown models — fine for
text-only fine-tunes; VLMs raise loudly to prevent silent image
dropping.
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import create_renderer_pool, RendererPool

from torchtitan.config import Configurable

__all__ = ["RendererConfig"]


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    name: str = "auto"
    """Renderer name (``"qwen3"`` / ``"gpt-oss"`` / ``"glm-5"`` / …) or
    ``"auto"`` to detect from the tokenizer's ``name_or_path``."""

    tool_parser: str | None = None
    """Override tool parser; only consumed by ``DefaultRenderer``.
    Hand-coded renderers (qwen3, etc.) ship their own tool parsing."""

    reasoning_parser: str | None = None
    """Override reasoning parser; only consumed by ``DefaultRenderer``."""

    preserve_all_thinking: bool = False
    """Keep ``reasoning_content`` on historical assistants the template
    would otherwise drop. Useful when the trainer learns from earlier
    thinking; off by default for token-parity with the chat template."""

    pool_size: int = 16
    """Number of independent tokenizer slots in the pool. Defaults to
    16 — enough headroom for N concurrent rollouts at a moderate
    group_size. Raise if you see event-loop stalls during multi-turn
    rollouts at high concurrency."""

    def build(self, tokenizer_path: str) -> RendererPool:
        """Construct a :class:`RendererPool` bound to ``tokenizer_path``."""
        return create_renderer_pool(
            tokenizer_path,
            renderer=self.name,
            size=self.pool_size,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            preserve_all_thinking=self.preserve_all_thinking,
        )
