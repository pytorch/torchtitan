#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test bitwise parity between the SGLang generator and the TorchTitan trainer.

  1. test_trainer_vs_sglang_decode:
      SGLang decode logprobs == trainer prefill logprobs of [prompt + completion]
      at the completion's positions, for completions sampled at temperature 1.
  2. test_sglang_decode_batch_invariance:
      A seeded request decodes the same tokens with the same logprobs alone and in
      a batch of mixed-length requests, after its prompt is cached.
  3. test_trainer_vs_sglang_decode_after_cached_turn:
      Test 1 for each session's next turn (its prompt extends the previous turn's
      prompt and completion), which resumes the session's cached prefix: KV, and
      the recurrent state of GDN and KDA layers.
  4. test_trainer_vs_sglang_decode_after_model_state_dict_load:
      Test 1 after perturbed trainer weights are loaded into SGLang in place (CUDA
      graphs kept) and requests move to the next policy version's cache salt.
  5. test_sglang_decode_across_model_state_dict_load:
      Loading the same weights while requests decode leaves their tokens and
      logprobs unchanged.
  6. test_prefill_replays_cuda_graphs:
      With prefill CUDA graphs ("breakable" or "full"), a prefill of new prompts
      replays one, rather than falling back to eager.
  7. test_context_length_limits (TestSGLangBitwiseParityVarlen only):
      As in vLLM, a prompt of max_context_length - 1 tokens runs and stops at
      max_context_length tokens in total, and a prompt of max_context_length
      tokens is rejected.

Debug models with random weights start from the trainer's weights, loaded through the
same in-place model state dict load.

Runs in the B200 RL CI job.
Run each class in its own launch, with as many processes as its DP x TP degree (one
for TestSGLangBitwiseParityKDA, four for TestSGLangBitwiseParityMoEDPEP,
TestSGLangBitwiseParityKDADPEP and TestSGLangBitwiseParityDistMoEDPEP, else two),
with the SGLang environment (see
torchtitan/rl/generator/sglang/README.md; the Dist-MoE classes also need the dist_moe
package):
    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_sglang_bitwise_parity.py::TestSGLangBitwiseParityVarlen -v
"""

import dataclasses
import functools
import importlib.util
import os
import shutil
import tempfile
import time
import unittest

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import CommConfig, TORCH_DTYPE_MAP
from torchtitan.config.transform import apply_transforms
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_model_config
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_model_config,
)
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, FinishedRequest, SamplingConfig
from torchtitan.rl.rollout.types import Prompt
from torchtitan.tools.utils import activate_cuda_flash_attention_impl
from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
)
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_kimi_k3_debug_varlen_batch_invariant,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    build_trainer_model,
    compute_trainer_prefill_logprobs,
    kimi_k3_debug_bitwise_config,
    make_prompt_tokens,
    qwen3_5_debug_bitwise_config,
)

# Each class runs under its own torchrun launch (see the module docstring).
pytestmark = pytest.mark.multi_gpu


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    dist.is_initialized() or "RANK" in os.environ,
    "requires torchrun launcher",
)
class SGLangBitwiseParityTestBase(unittest.TestCase):
    __test__ = False
    config_fn = None
    # Load the trainer's weights into the generator before the tests, for debug
    # models whose random weights each side would initialize differently.
    sync_weights_from_trainer = False
    # CUDA graphs SGLang captures (SGLangEngine.Config.cuda_graph).
    cuda_graph = "full"
    # SGLangEngine.Config.extra_server_args.
    extra_server_args = {}
    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50
    # Tokens a cached turn's prompt adds after the previous turn's completion. Even,
    # which keeps that turn's packed batch even: the trainer's MoE at EP 2 splits it
    # evenly over TP 2.
    NEXT_TURN_SUFFIX_LENGTH = 8
    # The policy version of the engine's weights, which salts the requests' KV.
    policy_version = 0
    # Off by default, so test 4 covers the cache salt.
    reset_kv_cache_on_model_state_dict_load = False
    # SGLang's chunked prefill budget (Engine max_num_batched_tokens); None: its default.
    max_num_batched_tokens = None

    @classmethod
    def setUpClass(cls):
        from torchtitan.rl.generator.sglang.engine import SGLangEngine
        from transformers import AutoTokenizer

        config = cls.config_fn()
        hf_path = os.environ.get("HF_ASSETS_PATH")
        if hf_path:
            config.hf_assets_path = hf_path
        activate_cuda_flash_attention_impl()
        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.trainer.debug.batch_invariant)
        if not dist.is_initialized():
            base_folder = tempfile.mkdtemp(prefix="rl_sglang_bitwise_")
            cls.addClassCleanup(shutil.rmtree, base_folder, ignore_errors=True)
            dist_utils.init_distributed(CommConfig(), base_folder=base_folder)

        cls.precision = config.precision
        cls.max_context_length = config.generator.model.max_context_length
        cls.model, cls.device, cls.parallelism_context = build_trainer_model(config)
        # Same workaround as test_bitwise_parity: the trainer and the generator
        # build equal but separate meshes in this process.
        torch._C._clear_DTensor_sharding_propagator_cache()
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()

        cls.engine = SGLangEngine.Config(
            cuda_graph=cls.cuda_graph, extra_server_args=cls.extra_server_args
        ).build(
            model=config.generator.model,
            parallelism=config.generator.parallelism,
            model_path=config.hf_assets_path,
            max_num_seqs=cls.BATCH_SIZE,
            precision=config.precision,
            # Leave room for the trainer model in this process.
            gpu_memory_limit=0.5,
            max_num_batched_tokens=cls.max_num_batched_tokens,
            checkpointer=(
                None
                if cls.sync_weights_from_trainer
                else CheckpointManager.Config(
                    initial_load_in_hf=True,
                    initial_load_path=config.hf_assets_path,
                )
            ),
            dist_moe_runtime=config.generator.dist_moe_runtime,
            # SGLang samples with the requests' seeds only with DebugConfig.seed.
            debug=dataclasses.replace(config.generator.debug, seed=0),
            reset_kv_cache_on_model_state_dict_load=(
                cls.reset_kv_cache_on_model_state_dict_load
            ),
            model_state_dict_load_mode="gpu_fetch",
            vision_processor=None,
            mooncake_master_address=None,
            rank=dist.get_rank(),
            generator_name="generator",
            output_dir=tempfile.gettempdir(),
        )
        if cls.sync_weights_from_trainer:
            _load_trainer_weights(cls.engine, cls.model, cls.precision)
        tokenizer = AutoTokenizer.from_pretrained(config.hf_assets_path)
        cls.prompt_ids = make_prompt_tokens(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, tokenizer
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.barrier()

    def _generate(
        self, prompt_ids: list[list[int]], *, load_at_step: int = -1
    ) -> list[FinishedRequest]:
        """Sample a completion of each prompt at temperature 1, each with its own seed
        and session, and return the results in prompt order. Equal prompts form one
        rollout group.

        With ``load_at_step`` >= 0, load the trainer's weights before that step.
        """
        requests = []
        for i, ids in enumerate(prompt_ids):
            request = EngineRequest(
                request_id=f"{time.monotonic_ns()}-{i}",
                prompt=Prompt(token_ids=ids),
                sampling=SamplingConfig(
                    temperature=1.0,
                    max_tokens=self.MAX_GEN_TOKENS,
                    seed=1000 + i,
                    stop_token_ids=[],
                ),
                group_id=prompt_ids.index(ids),
                routing_session_id=str(i),
                arrival_time=time.time(),
            )
            request.min_policy_version = self.policy_version
            requests.append(request)
        self.assertEqual(self.engine.add_requests(requests), [])
        finished = {}
        num_steps = 0
        while self.engine.has_unfinished_requests():
            if num_steps == load_at_step:
                _load_trainer_weights(self.engine, self.model, self.precision)
            for result in self.engine.step():
                finished[result.request_id] = result
            num_steps += 1
        self.assertGreater(num_steps, load_at_step)
        return [finished[request.request_id] for request in requests]

    def _assert_bitwise_equal(self, name, a, b) -> None:
        a = torch.as_tensor(a, dtype=torch.float32).cpu()
        b = torch.as_tensor(b, dtype=torch.float32).cpu()
        self.assertEqual(a.shape, b.shape, name)
        max_delta = (a - b).abs().max().item() if a.numel() else 0.0
        print(f"  {name}: max_delta={max_delta:.2e}, equal={torch.equal(a, b)}")
        self.assertTrue(
            torch.equal(a, b), f"{name}: NOT bitwise identical ({max_delta=:.2e})"
        )

    def _trainer_logprobs(self, token_ids):
        """``compute_trainer_prefill_logprobs`` of the trainer model."""
        with self.parallelism_context.activate_spmd(), torch.no_grad():
            return compute_trainer_prefill_logprobs(self.model, token_ids, self.device)

    def _check_trainer_vs_sglang_decode(
        self, prompt_ids: list[list[int]]
    ) -> list[FinishedRequest]:
        """Generate from ``prompt_ids`` and check the completions' logprobs against the
        trainer's prefill of prompt and completion; return the results."""
        results = self._generate(prompt_ids)
        sequences = [
            prompt + result.token_ids
            for prompt, result in zip(prompt_ids, results, strict=True)
        ]
        trainer_lps = self._trainer_logprobs(sequences)
        if dist.get_rank() == 0:
            for i, (prompt, result) in enumerate(zip(prompt_ids, results)):
                self._assert_bitwise_equal(
                    f"seq {i}: trainer prefill vs SGLang decode",
                    trainer_lps[i][len(prompt) - 1 :],
                    result.token_logprobs,
                )
        return results

    def test_trainer_vs_sglang_decode(self):
        # The last two requests share a prompt that no earlier test cached, like two
        # samples of a rollout group: the second must reuse the first's prompt KV.
        shared_prompt = self.prompt_ids[0][::-1]
        prompt_ids = self.prompt_ids[:-2] + [shared_prompt, shared_prompt]
        results = self._check_trainer_vs_sglang_decode(prompt_ids)
        if dist.get_rank() == 0:
            self.assertGreater(results[-1].metrics_inputs.num_cached_tokens, 0)

    def test_sglang_decode_batch_invariance(self):
        (single,) = self._generate(self.prompt_ids[:1])
        batched = self._generate(self.prompt_ids)[0]
        if dist.get_rank() == 0:
            self.assertEqual(single.token_ids, batched.token_ids)
            self._assert_bitwise_equal(
                "seq 0: SGLang decode(bsz=1) vs decode(bsz=5)",
                single.token_logprobs,
                batched.token_logprobs,
            )

    def test_trainer_vs_sglang_decode_after_cached_turn(self):
        results = self._generate(self.prompt_ids)
        # Each next turn: the history, then a few tokens standing in for a tool result.
        next_prompt_ids = [
            prompt + result.token_ids + prompt[: self.NEXT_TURN_SUFFIX_LENGTH]
            for prompt, result in zip(self.prompt_ids, results, strict=True)
        ]
        next_results = self._check_trainer_vs_sglang_decode(next_prompt_ids)
        if dist.get_rank() == 0:
            for prompt, result in zip(self.prompt_ids, next_results):
                # The cache covers the previous turn's completion too.
                self.assertGreater(result.metrics_inputs.num_cached_tokens, len(prompt))

    def test_sglang_decode_across_model_state_dict_load(self):
        # The trainer's weights are still the ones the generator loaded, and step 10
        # is mid-decode: prefill takes a step or two of the 50 tokens.
        expected = self._generate(self.prompt_ids)
        results = self._generate(self.prompt_ids, load_at_step=10)
        if dist.get_rank() == 0:
            for i, (result, expected_result) in enumerate(zip(results, expected)):
                self.assertEqual(result.token_ids, expected_result.token_ids)
                self._assert_bitwise_equal(
                    f"seq {i}: decode with vs without a load",
                    result.token_logprobs,
                    expected_result.token_logprobs,
                )

    def test_prefill_replays_cuda_graphs(self):
        # The other tests would pass with eager prefill too.
        if self.cuda_graph not in ("breakable", "full"):
            self.skipTest(f"cuda_graph={self.cuda_graph!r} captures no prefill graphs")
        model_runner = self.engine._scheduler.tp_worker.model_runner
        backend = model_runner.prefill_cuda_graph_runner.backend
        replay = backend.replay
        num_replays = 0

        def counting_replay(*args, **kwargs):
            nonlocal num_replays
            num_replays += 1
            return replay(*args, **kwargs)

        backend.replay = counting_replay
        try:
            # Prompts no earlier test cached, so prefill runs over all of them.
            self._generate([prompt[::-1] for prompt in self.prompt_ids])
        finally:
            backend.replay = replay
        self.assertGreater(num_replays, 0)

    def test_trainer_vs_sglang_decode_after_model_state_dict_load(self):
        # Runs last (unittest orders tests by name): it changes both models' weights.
        logprobs_before = self._trainer_logprobs(self.prompt_ids[0])
        with torch.no_grad():
            for param in self.model.parameters():
                param.mul_(1.01)
        self.assertFalse(
            torch.equal(self._trainer_logprobs(self.prompt_ids[0]), logprobs_before)
        )
        _load_trainer_weights(self.engine, self.model, self.precision)
        type(self).policy_version += 1
        self._check_trainer_vs_sglang_decode(self.prompt_ids)


def _load_trainer_weights(engine, model, precision) -> None:
    """Load the trainer model's weights into the engine, as a model state dict load."""
    trainer_state_dict = model.state_dict()
    # The trainer sends parameters in the generator's compute dtype, and buffers and
    # preserve_dtype_params in their own.
    own_dtype_keys = {name for name, _ in model.named_buffers()} | {
        name for name, _ in model.named_preserve_dtype_params()
    }
    param_dtype = TORCH_DTYPE_MAP[precision.param_dtype]

    def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
        with torch.no_grad():
            for key, tensor in state_dict.items():
                value = trainer_state_dict[key]
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                sent_dtype = value.dtype if key in own_dtype_keys else param_dtype
                assert tensor.dtype == sent_dtype, (key, tensor.dtype, sent_dtype)
                value = value.to(sent_dtype)
                if isinstance(tensor, DTensor):
                    value = distribute_tensor(
                        value, tensor.device_mesh, tensor.placements
                    )
                tensor.copy_(value)
        return list(state_dict)

    engine.load_model_state_dict(fetch)


class TestSGLangBitwiseParityVarlen(SGLangBitwiseParityTestBase):
    """Qwen3-0.6B, trainer TP 2 and generator TP 2."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_0_6b_varlen_batch_invariant)

    def test_context_length_limits(self):
        lengths = [self.max_context_length - 1, self.max_context_length]
        requests = [
            EngineRequest(
                request_id=f"{time.monotonic_ns()}-{length}",
                prompt=Prompt(token_ids=self.prompt_ids[0][:1] * length),
                sampling=SamplingConfig(
                    temperature=1.0, max_tokens=self.MAX_GEN_TOKENS, stop_token_ids=[]
                ),
                group_id=i,
                routing_session_id=f"context-{i}",
                arrival_time=time.time(),
            )
            for i, length in enumerate(lengths)
        ]
        for request in requests:
            request.min_policy_version = self.policy_version
        rejected = self.engine.add_requests(requests)
        self.assertEqual(
            [request_id for request_id, _ in rejected], [requests[1].request_id]
        )
        finished = []
        while self.engine.has_unfinished_requests():
            finished.extend(self.engine.step())
        self.assertEqual(
            [result.request_id for result in finished], [requests[0].request_id]
        )
        self.assertEqual(len(finished[0].token_ids), 1)


def _qwen3_moe_debug_tp2_ep2_config() -> Controller.Config:
    """The Qwen3 MoE debug model with EP 2 over TP 2, on both sides."""
    config = rl_grpo_qwen3_moe_debug_varlen_batch_invariant()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
            expert_parallel_degree=2,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=InferenceParallelismConfig(
            tensor_parallel_degree=2, expert_parallel_degree=2
        ),
    )
    return config


class TestSGLangBitwiseParityMoEEP(SGLangBitwiseParityTestBase):
    """Qwen3 MoE debug model with random weights, EP 2 over TP 2."""

    __test__ = True
    config_fn = staticmethod(_qwen3_moe_debug_tp2_ep2_config)
    sync_weights_from_trainer = True
    # The all-to-all token dispatcher syncs the host for its split sizes.
    cuda_graph = "none"


class TestSGLangBitwiseParityMoEDPEP(SGLangBitwiseParityTestBase):
    """Qwen3 MoE debug model with random weights, DP 2 x TP 2 with EP 4 across the DP
    ranks (SGLang's attention DP), on 4 GPUs. The base tests give both DP ranks the
    same requests; test_dp_ranks_with_different_requests gives them different ones."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_moe_debug_varlen_batch_invariant)
    sync_weights_from_trainer = True
    # The all-to-all token dispatcher syncs the host for its split sizes.
    cuda_graph = "none"

    def test_dp_ranks_with_different_requests(self):
        # DP rank 1 finishes first and runs idle steps while DP rank 0 decodes; each
        # DP rank's results still match the trainer.
        dp_rank = self.engine.data_parallel_rank
        self.assertEqual(self.engine.data_parallel_size, 2)
        prompt_ids = self.prompt_ids if dp_rank == 0 else self.prompt_ids[:1]
        results = self._generate(prompt_ids)
        sequences = [
            prompt + result.token_ids
            for prompt, result in zip(prompt_ids, results, strict=True)
        ]
        trainer_lps = self._trainer_logprobs(sequences)
        for i, (prompt, result) in enumerate(zip(prompt_ids, results)):
            self._assert_bitwise_equal(
                f"DP rank {dp_rank} seq {i}: trainer prefill vs SGLang decode",
                trainer_lps[i][len(prompt) - 1 :],
                result.token_logprobs,
            )


class TestSGLangBitwiseParityGDN(SGLangBitwiseParityTestBase):
    """Qwen3.5 debug model with random weights: GDN state slots across prefill, decode
    and cached prefixes."""

    __test__ = True
    config_fn = staticmethod(qwen3_5_debug_bitwise_config)
    # SGLang cannot capture full prefill graphs of linear attention models.
    cuda_graph = "breakable"
    sync_weights_from_trainer = True
    # Tests 4 and 5 then cover the reset path (retract, flush, continue).
    reset_kv_cache_on_model_state_dict_load = True

    def test_runs_on_unified_memory(self):
        """The engine runs GDN models on SGLang's unified pool, not on separate KV
        and state pools."""
        from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator

        self.assertIsInstance(
            self.engine._scheduler.token_to_kv_pool_allocator, ChainAllocator
        )


class TestSGLangBitwiseParityGDNHiCache(TestSGLangBitwiseParityGDN):
    """``TestSGLangBitwiseParityGDN`` with HiCache backing KV and GDN states up to
    host memory."""

    extra_server_args = {"enable_hierarchical_cache": True, "hicache_size": 8}

    def test_trainer_vs_sglang_decode_after_host_load_back(self):
        """Test 3 with the previous turns evicted from GPU memory, so the next turns
        load their KV and GDN states back from host memory into the unified pool."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        results = self._generate(self.prompt_ids)
        scheduler = self.engine._scheduler
        tree_cache = scheduler.tree_cache
        with scheduler.device_module.StreamContext(scheduler.schedule_stream):
            # Wait for the host backups, then drop every GPU copy.
            tree_cache.writing_check(write_back=True)
            tree_cache.evict(EvictParams(num_tokens=10**9, mamba_num=10**9))
        self.assertEqual(tree_cache.total_size(), (0, 0))
        next_prompt_ids = [
            prompt + result.token_ids + prompt[: self.NEXT_TURN_SUFFIX_LENGTH]
            for prompt, result in zip(self.prompt_ids, results, strict=True)
        ]
        # SGLang's request objects, for how many cached tokens came from the host.
        admitted = []
        admit = self.engine._output_collector.admit

        def record_admit(req, arrival_time):
            admitted.append(req)
            admit(req, arrival_time)

        self.engine._output_collector.admit = record_admit
        try:
            next_results = self._check_trainer_vs_sglang_decode(next_prompt_ids)
        finally:
            self.engine._output_collector.admit = admit
        if dist.get_rank() == 0:
            for prompt, req, result in zip(self.prompt_ids, admitted, next_results):
                num_cached_tokens = result.metrics_inputs.num_cached_tokens
                self.assertGreater(num_cached_tokens, len(prompt))
                self.assertEqual(req.cached_tokens_host, num_cached_tokens)


def _debug_model_config(
    build_model_config, *, expert_parallel_degree: int = 1, **kwargs
) -> Controller.Config:
    """A model's debug flavor at TP 2 on both sides, in the batch-invariant Qwen3 MoE
    debug recipe."""
    config = rl_grpo_qwen3_moe_debug_varlen_batch_invariant()
    trainer_parallelism = dataclasses.replace(
        config.trainer.parallelism,
        data_parallel_shard_degree=1,
        expert_parallel_degree=expert_parallel_degree,
    )
    generator_parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=2, expert_parallel_degree=expert_parallel_degree
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "debugmodel",
        seq_len=2048,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        **kwargs,
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (trainer_model, generator_model):
        model_config.local_compile.regions = []
    config.trainer = dataclasses.replace(
        config.trainer, model=trainer_model, parallelism=trainer_parallelism
    )
    config.generator = dataclasses.replace(
        config.generator, model=generator_model, parallelism=generator_parallelism
    )
    return config


class TestSGLangBitwiseParitySlidingWindow(SGLangBitwiseParityTestBase):
    """Muse Glimmer debug model with random weights, 128-token windows: of the 60- to
    150-token prompts with 50 generated tokens, one exceeds a window in prefill and
    three during decode."""

    __test__ = True
    config_fn = staticmethod(
        functools.partial(
            _debug_model_config,
            build_muse_glimmer_model_config,
            # Its lm_head soft-caps the logits, which the fp32 lm_head does not.
            fp32_lm_head=False,
        )
    )
    sync_weights_from_trainer = True


class TestSGLangBitwiseParityMLA(SGLangBitwiseParityTestBase):
    """DeepSeek V3 debug model with random weights: MLA, whose per-head K/V SGLang
    caches, with EP 2 over TP 2."""

    __test__ = True
    config_fn = staticmethod(
        functools.partial(
            _debug_model_config,
            build_deepseek_v3_model_config,
            expert_parallel_degree=2,
        )
    )
    sync_weights_from_trainer = True
    # The all-to-all token dispatcher syncs the host for its split sizes.
    cuda_graph = "none"


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0),
    "Batch-invariant Attention Gym KDA uses fixed-tile TMA kernels (SM90+)",
)
class TestSGLangBitwiseParityKDA(SGLangBitwiseParityTestBase):
    """Kimi K3 debug model with random weights: KDA state slots, with Attention Gym's
    replay state of each slot's unfinished 64-token chunk, and MLA. One GPU, as
    test_bitwise_parity's vLLM Kimi K3 class. This class covers KDA decode with replay
    state in CUDA graphs; TestSGLangBitwiseParityKDADPEP covers the TP-sharded replay
    state, without CUDA graphs.
    """

    __test__ = True
    config_fn = staticmethod(kimi_k3_debug_bitwise_config)
    # SGLang cannot capture full prefill graphs of linear attention models.
    cuda_graph = "breakable"
    sync_weights_from_trainer = True
    # Chunked prefill off the 64-token grid: SGLang's prefix cache then snapshots KDA
    # states mid-chunk, with replay state (TorchTitanKDAAttnBackend).
    max_num_batched_tokens = 100

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The RL trainer's forward runs in training mode (as test_bitwise_parity's
        # KimiK3BitwiseParityTestBase); for the MoE, it only adds load-balancing
        # statistics.
        cls.model.train()


def _kimi_k3_debug_dp2_tp2_ep4_config() -> Controller.Config:
    """The batch-invariant Kimi K3 debug recipe (KDA, MLA and MoE layers) at
    DP 2 x TP 2, EP 4, on both sides."""
    config = rl_grpo_kimi_k3_debug_varlen_batch_invariant()
    trainer_parallelism = dataclasses.replace(
        config.trainer.parallelism,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=2, tensor_parallel_degree=2, expert_parallel_degree=4
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_kimi_k3_model_config,
        "debugmodel",
        seq_len=config.trainer.model.max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (trainer_model, generator_model):
        model_config.local_compile.regions = []
    config.trainer = dataclasses.replace(
        config.trainer, model=trainer_model, parallelism=trainer_parallelism
    )
    config.generator = dataclasses.replace(
        config.generator, model=generator_model, parallelism=generator_parallelism
    )
    return config


class TestSGLangBitwiseParityKDADPEP(
    TestSGLangBitwiseParityKDA, TestSGLangBitwiseParityMoEDPEP
):
    """The Kimi K3 debug model at DP 2 x TP 2, EP 4 across the DP ranks, on 4 GPUs:
    KDA's TP-sharded replay state, and an idle DP rank's linear attention steps."""

    config_fn = staticmethod(_kimi_k3_debug_dp2_tp2_ep4_config)
    # The all-to-all token dispatcher syncs the host for its split sizes.
    cuda_graph = "none"


def _with_dist_moe(config: Controller.Config) -> Controller.Config:
    """``config`` with Dist-MoE experts on both sides."""
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    # The most rows a rank can receive: every EP rank's tokens.
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(
        scratch_capacity_factor=float(
            config.generator.parallelism.expert_parallel_degree
        ),
        inference=True,
    )
    config.trainer = apply_transforms(config.trainer, [DistMoeTransform()])
    config.generator = apply_transforms(config.generator, [DistMoeTransform()])
    return config


def _qwen3_moe_debug_dist_moe_tp2_ep2_config() -> Controller.Config:
    """``_qwen3_moe_debug_tp2_ep2_config`` with Dist-MoE experts on both sides."""
    return _with_dist_moe(_qwen3_moe_debug_tp2_ep2_config())


def _qwen3_moe_debug_dist_moe_dp2_tp2_ep4_config() -> Controller.Config:
    """The Qwen3 MoE debug recipe (DP 2 x TP 2, EP 4) with Dist-MoE experts on both
    sides."""
    return _with_dist_moe(rl_grpo_qwen3_moe_debug_varlen_batch_invariant())


@unittest.skipUnless(
    importlib.util.find_spec("dist_moe") is not None,
    "requires the optional dist_moe package",
)
class TestSGLangBitwiseParityDistMoE(SGLangBitwiseParityTestBase):
    """Dist-MoE experts, EP 2 over TP 2, with SGLang's CUDA graphs.

    The trainer runs forward only, with an inference runtime of its own, which takes
    no parameters that require gradients.
    """

    __test__ = True
    config_fn = staticmethod(_qwen3_moe_debug_dist_moe_tp2_ep2_config)
    sync_weights_from_trainer = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.requires_grad_(False)
        cls.trainer_dist_moe_runtime = cls.config_fn().generator.dist_moe_runtime.build(
            model_parts=[cls.model],
            parallelism_context=cls.parallelism_context,
            device=cls.device,
            # The longest trainer batch: test_trainer_vs_sglang_decode_after_cached_turn's.
            num_tokens_per_microbatch_per_dp_rank=cls.BATCH_SIZE
            * (
                cls.PROMPT_LENGTH + 2 * cls.MAX_GEN_TOKENS + cls.NEXT_TURN_SUFFIX_LENGTH
            ),
            pp_schedule=None,
        )

    def _trainer_logprobs(self, token_ids):
        sequences = token_ids if isinstance(token_ids[0], list) else [token_ids]
        num_tokens = sum(len(sequence) for sequence in sequences)
        # Every EP rank passes the same count. With DP, the EP group spans the DP
        # ranks, whose batches can differ in size: take the largest.
        num_local_tokens = torch.tensor(
            [num_tokens // self.parallelism_context.tp], device=self.device
        )
        dist.all_reduce(num_local_tokens, op=dist.ReduceOp.MAX)
        self.trainer_dist_moe_runtime.set_num_local_input_tokens_per_call(
            int(num_local_tokens.item())
        )
        return super()._trainer_logprobs(token_ids)


class TestSGLangBitwiseParityDistMoEDPEP(
    TestSGLangBitwiseParityDistMoE, TestSGLangBitwiseParityMoEDPEP
):
    """Dist-MoE experts at DP 2 x TP 2, EP 4 across the DP ranks, with SGLang's CUDA
    graphs, on 4 GPUs: covers the Dist-MoE count of the largest DP rank and an idle
    DP rank's steps in CUDA graphs."""

    config_fn = staticmethod(_qwen3_moe_debug_dist_moe_dp2_tp2_ep4_config)
    cuda_graph = "full"
