# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import time
from collections import deque

import pytest
import torch

from torchtitan.rl.observability.vllm import VllmPreemptionStatLogger
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

_BLOCK_SIZE = 16
_PROMPT_LEN = 32


def _build_scheduler(tmp_path, scheduler_cls, num_blocks):
    """vLLM's real scheduler over a fake KV pool; it reads the model config but loads no weights."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 128,
            }
        )
    )
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, max_model_len=2048
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=2048,
            is_encoder_decoder=False,
            async_scheduling=scheduler_cls is AsyncScheduler,
        ),
        cache_config=CacheConfig(block_size=_BLOCK_SIZE, enable_prefix_caching=True),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    kv_cache_spec = FullAttentionSpec(
        block_size=_BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
        ),
        block_size=_BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_logs_the_schedulers_preemptions_and_reprefilled_tokens(
    tmp_path, caplog, scheduler_cls
):
    """24 requests of 132-428 tokens share 85 blocks (1,360 tokens), so vLLM preempts.

    Ground truth comes from the scheduler: its preempted request ids, and the tokens it
    schedules beyond prompt + max_tokens - 1 per request (one prefill, then one decode per
    later token), which only recomputation adds.
    """
    scheduler = _build_scheduler(tmp_path, scheduler_cls, num_blocks=85)
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)
    stat_logger = VllmPreemptionStatLogger(scheduler.vllm_config, engine_index=3)

    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(_BLOCK_SIZE, sha256)
    num_tokens_without_preemption = 0
    for i in range(24):
        max_tokens = 100 + (i * 37) % 300
        num_tokens_without_preemption += _PROMPT_LEN + max_tokens - 1
        request = EngineCoreRequest(
            request_id=str(i),
            prompt_token_ids=[i + 1] * _PROMPT_LEN,
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                ignore_eos=True,
                detokenize=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id=str(i),
        )
        output_processor.add_request(request, prompt=None)
        scheduler.add_request(Request.from_engine_core_request(request, block_hasher))

    # Same calls as LLMEngine.step, with every request sampling token 7. The 2,048-token step
    # budget exceeds the 1,360-token KV pool, so no prefill is split across steps.
    def run_model_and_record(scheduler_output):
        req_ids = list(scheduler_output.num_scheduled_tokens)
        engine_core_outputs = scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                sampled_token_ids=[[7] for _ in req_ids],
            ),
        )[0]
        iteration_stats = IterationStats() if engine_core_outputs.outputs else None
        output_processor.process_outputs(
            engine_core_outputs.outputs,
            engine_core_timestamp=engine_core_outputs.timestamp,
            iteration_stats=iteration_stats,
        )
        stat_logger.record(engine_core_outputs.scheduler_stats, iteration_stats)

    # AsyncScheduler schedules step k+1 before it reads step k's output (one step in flight).
    # Once nothing new is scheduled, read the remaining outputs.
    steps_in_flight = 2 if scheduler_cls is AsyncScheduler else 1
    in_flight = deque()
    num_preemptions = 0
    num_scheduled_tokens = 0
    while scheduler.has_unfinished_requests() or in_flight:
        scheduler_output = scheduler.schedule()
        num_preemptions += len(scheduler_output.preempted_req_ids)
        num_scheduled_tokens += scheduler_output.total_num_scheduled_tokens
        if scheduler_output.total_num_scheduled_tokens > 0:
            in_flight.append(scheduler_output)
        while len(in_flight) >= steps_in_flight or (
            in_flight and scheduler_output.total_num_scheduled_tokens == 0
        ):
            run_model_and_record(in_flight.popleft())
    num_reprefilled_tokens = num_scheduled_tokens - num_tokens_without_preemption
    assert num_preemptions > 10

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="torchtitan.rl.observability.vllm"):
        stat_logger.log()
        stat_logger.log()
    # endswith: the Monarch log filter that other tests install prefixes "[actor=...]".
    first_line, second_line = caplog.messages
    assert first_line.endswith(
        f"Engine 003: Preemptions: {num_preemptions}, "
        f"Re-prefilled tokens: {num_reprefilled_tokens}"
    )
    assert second_line.endswith("Engine 003: Preemptions: 0, Re-prefilled tokens: 0")
