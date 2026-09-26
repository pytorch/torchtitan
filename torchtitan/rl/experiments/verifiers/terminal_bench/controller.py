# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run eval-only Terminal-Bench without scheduling unused training rollouts."""

from dataclasses import dataclass

from torchtitan.rl.controller import Controller


class TerminalBenchController(Controller):
    @dataclass(kw_only=True, slots=True)
    class Config(Controller.Config):
        eval_only: bool = False

    async def run(self) -> None:
        if self.config.eval_only:
            await self._validate_and_log(step=self.start_step)
            return
        await super().run()
