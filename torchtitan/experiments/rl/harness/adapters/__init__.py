# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wire-format adapters: HTTP endpoints that serve the trained policy to an
external CLI agent in its native API dialect while capturing each turn as
on-policy training tokens. ``anthropic`` covers the Anthropic Messages API
(Claude Code); add an ``openai`` module for OpenAI Chat/Responses agents
(Codex/OpenCode) -- the per-turn token capture is shared."""

from torchtitan.experiments.rl.harness.adapters.anthropic import (
    AnthropicAdapter,
    CapturedTurn,
)

__all__ = ["AnthropicAdapter", "CapturedTurn"]
