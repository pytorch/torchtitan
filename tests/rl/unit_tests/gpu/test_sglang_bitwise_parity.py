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

Debug models with random weights start from the trainer's weights, loaded through the
same in-place model state dict load.

Runs in the B200 RL CI job.
Run each class in its own launch with the SGLang environment (see
torchtitan/rl/generator/sglang/README.md; the Dist-MoE class also needs the dist_moe
package):
    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_sglang_bitwise_parity.py::TestSGLangBitwiseParityVarlen -v
"""

import dataclasses
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
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.paged_attention import (
    activate_trainer_flash_attention_impl,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
)
from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
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
class SGLangBitwiseParityTestBase(unittest.TestCase):
    __test__ = False
    config_fn = None
    # Load the trainer's weights into the generator before the tests, for debug
    # models whose random weights each side would initialize differently.
    sync_weights_from_trainer = False
    # CUDA graphs SGLang captures (SGLangEngine.Config.cuda_graph).
    cuda_graph = "decode_and_prefill"
    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50
    # The policy version of the engine's weights, which salts the requests' KV.
    policy_version = 0

    @classmethod
    def setUpClass(cls):
        from torchtitan.rl.generator.sglang.engine import SGLangEngine
        from transformers import AutoTokenizer

        config = cls.config_fn()
        hf_path = os.environ.get("HF_ASSETS_PATH")
        if hf_path:
            config.hf_assets_path = hf_path
        activate_trainer_flash_attention_impl()
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

        cls.engine = SGLangEngine.Config(cuda_graph=cls.cuda_graph).build(
            model=config.generator.model,
            parallelism=config.generator.parallelism,
            model_path=config.hf_assets_path,
            max_num_seqs=cls.BATCH_SIZE,
            precision=config.precision,
            # Leave room for the trainer model in this process.
            gpu_memory_limit=0.5,
            max_num_batched_tokens=None,
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
            reset_kv_cache_on_model_state_dict_load=False,
            model_state_dict_load_mode="gpu_fetch",
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

    def _trainer_logprobs(self, token_ids):
        """``compute_trainer_prefill_logprobs`` of the trainer model."""
        with self.parallelism_context.activate_spmd(), torch.no_grad():
            return compute_trainer_prefill_logprobs(self.model, token_ids, self.device)

    def test_trainer_vs_sglang_decode(self):
        gen_ids, gen_lps = self._generate(self.prompt_ids)
        sequences = [p + g for p, g in zip(self.prompt_ids, gen_ids, strict=True)]
        trainer_lps = self._trainer_logprobs(sequences)
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
        logprobs_before = self._trainer_logprobs(self.prompt_ids[0])
        with torch.no_grad():
            for param in self.model.parameters():
                param.mul_(1.01)
        self.assertFalse(
            torch.equal(self._trainer_logprobs(self.prompt_ids[0]), logprobs_before)
        )
        _load_trainer_weights(self.engine, self.model, self.precision)
        type(self).policy_version += 1
        self.test_trainer_vs_sglang_decode()


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


def _qwen3_moe_debug_dist_moe_tp2_ep2_config() -> Controller.Config:
    """``_qwen3_moe_debug_tp2_ep2_config`` with Dist-MoE experts on both sides."""
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = _qwen3_moe_debug_tp2_ep2_config()
    # The most rows a rank can receive at EP 2.
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(
        scratch_capacity_factor=2.0, inference=True
    )
    config.trainer = apply_transforms(config.trainer, [DistMoeTransform()])
    config.generator = apply_transforms(config.generator, [DistMoeTransform()])
    return config


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
        cls.trainer_dist_moe_runtime = DistMoeRuntime.Config(
            scratch_capacity_factor=2.0, inference=True
        ).build(
            model_parts=[cls.model],
            parallelism_context=cls.parallelism_context,
            device=cls.device,
            num_tokens_per_microbatch_per_dp_rank=cls.BATCH_SIZE
            * (cls.PROMPT_LENGTH + cls.MAX_GEN_TOKENS),
            pp_schedule=None,
        )

    def _trainer_logprobs(self, token_ids):
        sequences = token_ids if isinstance(token_ids[0], list) else [token_ids]
        num_tokens = sum(len(sequence) for sequence in sequences)
        self.trainer_dist_moe_runtime.set_num_local_input_tokens_per_call(
            num_tokens // self.parallelism_context.tp
        )
        return super()._trainer_logprobs(token_ids)
