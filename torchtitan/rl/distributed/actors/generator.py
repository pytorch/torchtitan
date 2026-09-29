# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the standalone vLLM generator."""

from __future__ import annotations

from monarch.actor import Actor, Channel, concurrent_endpoint, context, current_rank

from torchtitan.config import CompileConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator import SamplingConfig, VLLMGenerator
from torchtitan.rl.types import Completion


class _GeneratorActorEndpoints:
    def __init__(
        self,
        config: VLLMGenerator.Config,
        *,
        model_config: Decoder.Config,
        model_path: str,
        compile_config: CompileConfig | None,
        max_num_seqs: int,
        output_dir: str,
    ) -> None:
        super().__init__(
            config,
            model_config=model_config,
            model_path=model_path,
            compile_config=compile_config,
            max_num_seqs=max_num_seqs,
            output_dir=output_dir,
            rank=current_rank().rank,
            generator_name=context().actor_instance.actor_id.actor_name,
            open_result_channel=Channel.open,
        )

    @concurrent_endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        await super().sync_log_step(step, relative_step)

    @concurrent_endpoint
    async def start_engine_loop(self) -> None:
        await super().start_engine_loop()

    @concurrent_endpoint
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        routing_session_id: str,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> Completion:
        return await super().generate(
            prompt_token_ids,
            request_id=request_id,
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
        )

    @concurrent_endpoint
    async def pull_model_state_dict(self, version: int) -> None:
        await super().pull_model_state_dict(version)

    @concurrent_endpoint
    async def close(self) -> None:
        await super().close()


class VLLMGeneratorActor(Actor, _GeneratorActorEndpoints, VLLMGenerator):
    """Expose a standalone :class:`VLLMGenerator` through Monarch endpoints."""
