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
  3. test_trainer_vs_sglang_decode_after_model_state_dict_load:
      Test 1 after perturbed trainer weights are loaded into SGLang in place (CUDA
      graphs kept) and requests move to the next policy version's cache salt.

Runs in the B200 RL CI job. Run with the SGLang environment (see
torchtitan/rl/generator/sglang/README.md):
    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_sglang_bitwise_parity.py -v
"""

import dataclasses
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
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan.tools.utils import activate_cuda_flash_attention_impl
from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
)

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    build_trainer_model,
    compute_trainer_prefill_logprobs,
    make_prompt_tokens,
)

# Each class runs under its own two-GPU torchrun launch.
pytestmark = pytest.mark.multi_gpu


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    dist.is_initialized() or "RANK" in os.environ,
    "requires torchrun launcher",
)
class TestSGLangBitwiseParityVarlen(unittest.TestCase):
    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50
    # The policy version of the engine's weights, which salts the requests' KV.
    policy_version = 0

    @classmethod
    def setUpClass(cls):
        from torchtitan.rl.generator.sglang.engine import SGLangEngine
        from transformers import AutoTokenizer

        config = rl_grpo_qwen3_0_6b_varlen_batch_invariant()
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
        cls.model, cls.device, cls.parallelism_context = build_trainer_model(config)
        # Same workaround as test_bitwise_parity: the trainer and the generator
        # build equal but separate meshes in this process.
        torch._C._clear_DTensor_sharding_propagator_cache()
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()

        cls.engine = SGLangEngine.Config().build(
            model=config.generator.model,
            parallelism=config.generator.parallelism,
            model_path=config.hf_assets_path,
            max_num_seqs=cls.BATCH_SIZE,
            precision=config.precision,
            # Leave room for the trainer model in this process.
            gpu_memory_limit=0.5,
            max_num_batched_tokens=None,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                initial_load_path=config.hf_assets_path,
            ),
            dist_moe_runtime=None,
            # SGLang samples with the requests' seeds only with DebugConfig.seed.
            debug=dataclasses.replace(config.generator.debug, seed=0),
            reset_kv_cache_on_model_state_dict_load=False,
            model_state_dict_load_mode="gpu_fetch",
            rank=dist.get_rank(),
            generator_name="generator",
            output_dir=tempfile.gettempdir(),
        )
        tokenizer = AutoTokenizer.from_pretrained(config.hf_assets_path)
        cls.prompt_ids = make_prompt_tokens(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, tokenizer
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.barrier()

    def _generate(self, prompt_ids: list[list[int]]):
        """Sample a completion of each prompt at temperature 1, each with its own seed.

        Returns ``(token_ids, logprobs)`` per prompt, in prompt order.
        """
        requests = []
        for i, ids in enumerate(prompt_ids):
            request = EngineRequest(
                request_id=f"{time.monotonic_ns()}-{i}",
                prompt_token_ids=ids,
                sampling=SamplingConfig(
                    temperature=1.0,
                    max_tokens=self.MAX_GEN_TOKENS,
                    seed=1000 + i,
                    stop_token_ids=[],
                ),
                group_id=i,
                routing_session_id=str(i),
                arrival_time=time.time(),
            )
            request.min_policy_version = self.policy_version
            requests.append(request)
        self.assertEqual(self.engine.add_requests(requests), [])
        finished = {}
        while self.engine.has_unfinished_requests():
            for result in self.engine.step():
                finished[result.request_id] = result
        results = [finished[request.request_id] for request in requests]
        return (
            [result.token_ids for result in results],
            [result.token_logprobs for result in results],
        )

    def _assert_bitwise_equal(self, name, a, b) -> None:
        a = torch.as_tensor(a, dtype=torch.float32).cpu()
        b = torch.as_tensor(b, dtype=torch.float32).cpu()
        self.assertEqual(a.shape, b.shape, name)
        max_delta = (a - b).abs().max().item() if a.numel() else 0.0
        print(f"  {name}: max_delta={max_delta:.2e}, equal={torch.equal(a, b)}")
        self.assertTrue(
            torch.equal(a, b), f"{name}: NOT bitwise identical ({max_delta=:.2e})"
        )

    def test_trainer_vs_sglang_decode(self):
        gen_ids, gen_lps = self._generate(self.prompt_ids)
        sequences = [p + g for p, g in zip(self.prompt_ids, gen_ids, strict=True)]
        with self.parallelism_context.activate_spmd(), torch.no_grad():
            trainer_lps = compute_trainer_prefill_logprobs(
                self.model, sequences, self.device
            )
        if dist.get_rank() == 0:
            for i, prompt in enumerate(self.prompt_ids):
                self._assert_bitwise_equal(
                    f"seq {i}: trainer prefill vs SGLang decode",
                    trainer_lps[i][len(prompt) - 1 :],
                    gen_lps[i],
                )

    def test_sglang_decode_batch_invariance(self):
        single_ids, single_lps = self._generate(self.prompt_ids[:1])
        batched_ids, batched_lps = self._generate(self.prompt_ids)
        if dist.get_rank() == 0:
            self.assertEqual(single_ids[0], batched_ids[0])
            self._assert_bitwise_equal(
                "seq 0: SGLang decode(bsz=1) vs decode(bsz=5)",
                single_lps[0],
                batched_lps[0],
            )

    def test_trainer_vs_sglang_decode_after_model_state_dict_load(self):
        # Runs last (unittest orders tests by name): it changes both models' weights.
        def trainer_logprobs() -> torch.Tensor:
            with self.parallelism_context.activate_spmd(), torch.no_grad():
                return compute_trainer_prefill_logprobs(
                    self.model, self.prompt_ids[0], self.device
                )

        logprobs_before = trainer_logprobs()
        with torch.no_grad():
            for param in self.model.parameters():
                param.mul_(1.01)
        self.assertFalse(torch.equal(trainer_logprobs(), logprobs_before))
        trainer_state_dict = self.model.state_dict()
        # The trainer sends parameters in the generator's compute dtype, and buffers
        # and preserve_dtype_params in their own.
        own_dtype_keys = {name for name, _ in self.model.named_buffers()} | {
            name for name, _ in self.model.named_preserve_dtype_params()
        }
        param_dtype = TORCH_DTYPE_MAP[self.precision.param_dtype]

        def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
            with torch.no_grad():
                for key, tensor in state_dict.items():
                    value = trainer_state_dict[key]
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    sent_dtype = value.dtype if key in own_dtype_keys else param_dtype
                    self.assertEqual(tensor.dtype, sent_dtype, key)
                    value = value.to(sent_dtype)
                    if isinstance(tensor, DTensor):
                        value = distribute_tensor(
                            value, tensor.device_mesh, tensor.placements
                        )
                    tensor.copy_(value)
            return list(state_dict)

        self.engine.load_model_state_dict(fetch)
        type(self).policy_version += 1
        self.test_trainer_vs_sglang_decode()
