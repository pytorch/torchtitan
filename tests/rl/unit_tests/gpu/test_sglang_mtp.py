#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MTP drafts in the SGLang generator (SGLangEngine.Config.num_speculative_tokens = 3),
checked against the TorchTitan trainer scoring the same tokens.

Drafts change the decode kernels (SGLang's verify kernels), so tokens and logprobs are
not bitwise equal to the trainer's; each test bounds the gap instead.

  1. test_sampled_tokens_follow_the_trainer (temperature 1): the drafts are accepted,
     verify steps replay CUDA graphs, the sampled tokens follow the trainer's
     distribution, and their logprobs are close to the trainer's.
  2. test_greedy_tokens_are_the_trainers_argmax: every greedy token is the trainer's
     argmax, or within a near tie of it.
  3. test_cached_turn_resumes_the_drafts_state: test 2 on each session's next turn,
     which resumes the GDN state SGLang's verify commit snapshotted.
  4. test_same_seed_same_tokens: a seeded request samples the same tokens twice.
  5. test_drafts_at_the_context_limit: a request whose drafts reach past
     max_context_length stops at it.
  6. test_retracted_requests_resume: test 2 for requests that weight loads retract
     mid-decode, which come back on fresh ReplaySSM rings.
  7. test_weight_load_reaches_the_draft (runs last: it changes both models' weights):
     test 2 after a model state dict load, and the draft still shares the model's
     embedding and LM head.

Runs in the B200 RL CI job. Needs a Qwen3.5-0.8B checkpoint
(TORCHTITAN_QWEN3_5_0_8B_HF_PATH), whose mtp.* weights the drafts use, and the SGLang
environment (torchtitan/rl/generator/sglang/README.md):
    TORCHTITAN_QWEN3_5_0_8B_HF_PATH=<checkpoint> torchrun --nproc_per_node=1 -m pytest \
        tests/rl/unit_tests/gpu/test_sglang_mtp.py -v
"""

import dataclasses
import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import CommConfig, DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, FinishedRequest, SamplingConfig
from torchtitan.rl.rollout.types import Prompt
from torchtitan.tools.utils import activate_cuda_flash_attention_impl
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_5_9b_varlen
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

from tests.rl.unit_tests.gpu.test_bitwise_parity import build_trainer_model
from tests.rl.unit_tests.gpu.test_sglang_bitwise_parity import load_trainer_weights

MODEL_ENV = "TORCHTITAN_QWEN3_5_0_8B_HF_PATH"
NUM_SPECULATIVE_TOKENS = 3


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MODEL_ENV in os.environ, f"set {MODEL_ENV}")
@unittest.skipUnless(
    dist.is_initialized() or "RANK" in os.environ, "requires torchrun launcher"
)
class TestSGLangMTP(unittest.TestCase):
    NUM_PROMPTS = 16
    PROMPT_LENGTH = 1024
    NEXT_TURN_SUFFIX_LENGTH = 8

    # Each bar leaves a margin over this test's numbers on one GB300. Measured 2.48
    # tokens per verify step. On a similar load, SGLang's default verify (argmax
    # drafts) gave 1.70 where this verify gave 2.37.
    MIN_TOKENS_PER_VERIFY_STEP = 2.0
    # Measured 0.014 at temperature 1; greedy 0.0021, and 0.0081 after the x1.01
    # weight load (0.0075 without drafts).
    MAX_MEAN_ABS_LOGPROB_DELTA_SAMPLED = 0.03
    MAX_MEAN_ABS_LOGPROB_DELTA_GREEDY = 0.01
    # The trainer's max logprob minus that of a greedy token. Measured at most 0.18
    # (0.14 without drafts): near ties that the trainer's and SGLang's kernels round
    # apart.
    MAX_GREEDY_ARGMAX_GAP = 0.25

    @classmethod
    def setUpClass(cls):
        from torchtitan.rl.generator.sglang.engine import SGLangEngine
        from transformers import AutoTokenizer

        checkpoint = os.environ[MODEL_ENV]
        activate_cuda_flash_attention_impl()
        if not dist.is_initialized():
            base_folder = tempfile.mkdtemp(prefix="rl_sglang_mtp_")
            cls.addClassCleanup(shutil.rmtree, base_folder, ignore_errors=True)
            dist_utils.init_distributed(CommConfig(), base_folder=base_folder)

        parallelism = ParallelismConfig(data_parallel_shard_degree=1)
        trainer_model, generator_model = build_rl_model_configs(
            build_model_config,
            "0.8B",
            trainer_parallelism=parallelism,
            generator_parallelism=InferenceParallelismConfig(),
            seq_len=4096,
        )
        config = rl_grpo_qwen3_5_9b_varlen()
        config = dataclasses.replace(
            config,
            hf_assets_path=checkpoint,
            trainer=dataclasses.replace(
                config.trainer, model=trainer_model, parallelism=parallelism
            ),
        )
        cls.precision = config.precision
        cls.max_context_length = generator_model.max_context_length
        cls.model, cls.device, cls.parallelism_context = build_trainer_model(config)

        cls.engine = SGLangEngine.Config(
            cuda_graph="breakable", num_speculative_tokens=NUM_SPECULATIVE_TOKENS
        ).build(
            model=generator_model,
            parallelism=InferenceParallelismConfig(),
            model_path=checkpoint,
            max_num_seqs=cls.NUM_PROMPTS,
            precision=config.precision,
            # Leave room for the trainer model in this process.
            gpu_memory_limit=0.5,
            max_num_batched_tokens=None,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True, initial_load_path=checkpoint
            ),
            dist_moe_runtime=None,
            # Seeds the drafts' sampling and the verify coins, so the bars below give
            # the same verdict on every run.
            debug=DebugConfig(seed=0),
            # A model state dict load then clears SGLang's cache and retracts running
            # requests (tests 4, 6 and 7).
            reset_kv_cache_on_model_state_dict_load=True,
            model_state_dict_load_mode="gpu_fetch",
            vision_processor=None,
            mooncake_master_address=None,
            rank=dist.get_rank(),
            generator_name="generator",
            output_dir=tempfile.gettempdir(),
        )

        # Real text: drafts guess repeated or random text too easily or too badly.
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        rl_root = Path(__file__).resolve().parents[4] / "torchtitan" / "rl"
        text = "\n".join(path.read_text() for path in sorted(rl_root.rglob("*.md")))
        cls.token_ids = tokenizer.encode(text)
        if len(cls.token_ids) < cls.NUM_PROMPTS * cls.PROMPT_LENGTH:
            raise RuntimeError(
                f"{rl_root}'s markdown has only {len(cls.token_ids)} tokens"
            )
        cls.prompt_ids = [
            cls.token_ids[i * cls.PROMPT_LENGTH : (i + 1) * cls.PROMPT_LENGTH]
            for i in range(cls.NUM_PROMPTS)
        ]

    def _generate(
        self,
        prompt_ids: list[list[int]],
        *,
        temperature: float,
        max_tokens: int,
        load_at_steps: tuple[int, ...] = (),
    ) -> list[FinishedRequest]:
        """One completion per prompt, each with its own seed and session, in prompt
        order. Loads the trainer's weights before each step in ``load_at_steps``."""
        requests = []
        for i, ids in enumerate(prompt_ids):
            request = EngineRequest(
                request_id=f"{time.monotonic_ns()}-{i}",
                prompt=Prompt(token_ids=ids),
                sampling=SamplingConfig(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=1000 + i,
                    stop_token_ids=[],
                ),
                group_id=i,
                routing_session_id=str(i),
                arrival_time=time.time(),
            )
            request.min_policy_version = 0
            requests.append(request)

        self.assertEqual(self.engine.add_requests(requests), [])
        finished = {}
        num_steps = 0
        while self.engine.has_unfinished_requests():
            if num_steps in load_at_steps:
                load_trainer_weights(self.engine, self.model, self.precision)
            for result in self.engine.step():
                finished[result.request_id] = result
            num_steps += 1
        self.assertGreater(num_steps, max(load_at_steps, default=-1))
        return [finished[request.request_id] for request in requests]

    def _trainer_scores(
        self, prompt_ids: list[int], result: FinishedRequest
    ) -> dict[str, torch.Tensor]:
        """The trainer's view of each completion token x: log p(x), the entropy of p,
        and max log p, for p the trainer's next-token distribution there."""
        sequence = torch.tensor(prompt_ids + result.token_ids, device=self.device)
        positions = torch.arange(len(sequence), device=self.device)
        with self.parallelism_context.activate_spmd(), torch.no_grad():
            logits = self.model(
                sequence,
                attention_metadata=self.model._get_attention_metadata(positions),
                positions=positions,
            )
        first = len(prompt_ids) - 1
        logprobs = torch.log_softmax(logits[first:-1].float(), dim=-1)
        token_logprobs = logprobs.gather(-1, sequence[first + 1 :, None])[:, 0]
        return {
            "token_logprobs": token_logprobs.cpu(),
            "entropy": -(logprobs.exp() * logprobs).sum(-1).cpu(),
            "max_logprobs": logprobs.max(-1).values.cpu(),
        }

    def _check_greedy(
        self, prompt_ids: list[list[int]], results: list[FinishedRequest]
    ) -> None:
        """Every greedy token is the trainer's argmax or within a near tie of it, and
        its logprob is close to the trainer's."""
        gaps, deltas = [], []
        for prompt, result in zip(prompt_ids, results, strict=True):
            scores = self._trainer_scores(prompt, result)
            gaps.append(scores["max_logprobs"] - scores["token_logprobs"])
            deltas.append(
                torch.tensor(result.token_logprobs) - scores["token_logprobs"]
            )
        gap, delta = torch.cat(gaps), torch.cat(deltas).abs()

        print(
            f"  greedy, {gap.numel()} tokens: argmax gap max {gap.max():.4f}, "
            f"logprob |delta| mean {delta.mean():.5f} max {delta.max():.4f}"
        )
        self.assertLessEqual(gap.max().item(), self.MAX_GREEDY_ARGMAX_GAP)
        self.assertLess(delta.mean().item(), self.MAX_MEAN_ABS_LOGPROB_DELTA_GREEDY)

    def test_sampled_tokens_follow_the_trainer(self):
        model_runner = self.engine._scheduler.tp_worker.model_runner
        graph_runner = model_runner.decode_cuda_graph_runner
        execute = graph_runner.execute
        num_verify_replays = 0

        def counting_execute(forward_batch, *args, **kwargs):
            nonlocal num_verify_replays
            num_verify_replays += forward_batch.forward_mode.is_target_verify()
            return execute(forward_batch, *args, **kwargs)

        graph_runner.execute = counting_execute
        try:
            results = self._generate(self.prompt_ids, temperature=1.0, max_tokens=512)
        finally:
            graph_runner.execute = execute
        self.assertGreater(num_verify_replays, 0)

        # The first token of each request comes from prefill.
        num_verify_tokens = sum(len(result.token_ids) - 1 for result in results)
        num_verify_steps = sum(
            result.metrics_inputs.num_verify_steps for result in results
        )
        tokens_per_verify_step = num_verify_tokens / num_verify_steps
        print(f"  tokens per verify step {tokens_per_verify_step:.2f}")
        self.assertGreater(tokens_per_verify_step, self.MIN_TOKENS_PER_VERIFY_STEP)

        # For x sampled from p, log p(x) + H(p) has mean 0. A sampler that favors
        # the draft's guesses (likelier tokens) raises it: greedy outputs score +0.25.
        statistics, deltas = [], []
        for prompt, result in zip(self.prompt_ids, results, strict=True):
            scores = self._trainer_scores(prompt, result)
            statistics.append(scores["token_logprobs"] + scores["entropy"])
            deltas.append(
                torch.tensor(result.token_logprobs) - scores["token_logprobs"]
            )
        statistic, delta = torch.cat(statistics), torch.cat(deltas).abs()

        standard_error = statistic.std().item() / math.sqrt(statistic.numel())
        print(
            f"  {statistic.numel()} tokens: log p + H mean {statistic.mean():+.4f} "
            f"(standard error {standard_error:.4f}); logprob |delta| mean "
            f"{delta.mean():.5f} max {delta.max():.4f}"
        )
        self.assertLess(abs(statistic.mean().item()), 3 * standard_error)
        self.assertLess(delta.mean().item(), self.MAX_MEAN_ABS_LOGPROB_DELTA_SAMPLED)
        # No token whose generator-to-trainer probability ratio leaves [0.2, 5].
        self.assertLess(delta.max().item(), math.log(5))

    def test_greedy_tokens_are_the_trainers_argmax(self):
        results = self._generate(self.prompt_ids, temperature=0.0, max_tokens=512)
        self._check_greedy(self.prompt_ids, results)

    def test_cached_turn_resumes_the_drafts_state(self):
        # Reversed: prompts no other test sends, so turn 1 caches them first.
        prompt_ids = [prompt[::-1] for prompt in self.prompt_ids[:8]]
        results = self._generate(prompt_ids, temperature=0.0, max_tokens=128)
        # Each next turn: the history, then a few tokens standing in for a tool result.
        next_prompt_ids = [
            prompt + result.token_ids + prompt[: self.NEXT_TURN_SUFFIX_LENGTH]
            for prompt, result in zip(prompt_ids, results, strict=True)
        ]
        next_results = self._generate(next_prompt_ids, temperature=0.0, max_tokens=128)
        for prompt, result in zip(prompt_ids, next_results, strict=True):
            # The cache covers the previous turn's completion too.
            self.assertGreater(result.metrics_inputs.num_cached_tokens, len(prompt))
        self._check_greedy(next_prompt_ids, next_results)

    def test_same_seed_same_tokens(self):
        runs = []
        for _ in range(2):
            # A load of the same weights clears SGLang's cache
            # (reset_kv_cache_on_model_state_dict_load), so both runs prefill alike.
            load_trainer_weights(self.engine, self.model, self.precision)
            runs.append(
                self._generate(self.prompt_ids[:4], temperature=1.0, max_tokens=128)
            )
        for first, second in zip(*runs, strict=True):
            self.assertEqual(first.token_ids, second.token_ids)

    def test_drafts_at_the_context_limit(self):
        # Prefill emits the token at max_context_length - 2; the verify step after it
        # runs drafts at max_context_length and max_context_length + 1. SGLang's context
        # and the engine's RoPE caches are CONTEXT_LENGTH_MARGIN tokens longer, and
        # tokens past the limit are cut.
        prompt = self.token_ids[-(self.max_context_length - 2) :]
        (result,) = self._generate([prompt], temperature=1.0, max_tokens=64)
        self.assertEqual(len(result.token_ids), 2)
        self.assertGreater(result.metrics_inputs.num_verify_steps, 0)

    def test_retracted_requests_resume(self):
        # RL loads weights while requests decode. Each load retracts the running
        # requests (reset_kv_cache_on_model_state_dict_load), which then prefill again
        # on new rows with empty ReplaySSM rings. Reversed: prompts no other test sends.
        prompt_ids = [prompt[::-1] for prompt in self.prompt_ids[8:16]]
        results = self._generate(
            prompt_ids, temperature=0.0, max_tokens=128, load_at_steps=(10, 22)
        )
        self.assertGreater(
            min(result.metrics_inputs.num_preemptions for result in results), 0
        )
        self._check_greedy(prompt_ids, results)

    def test_weight_load_reaches_the_draft(self):
        # Runs last (unittest orders tests by name): it changes both models' weights.
        with torch.no_grad():
            for param in self.model.parameters():
                param.mul_(1.01)
        load_trainer_weights(self.engine, self.model, self.precision)

        decoder = self.engine._get_model().model.decoder
        draft = self.engine._scheduler.draft_worker.draft_worker.draft_runner.model
        self.assertEqual(
            draft.model.embed_tokens.weight.data_ptr(),
            decoder.tok_embeddings.weight.data_ptr(),
        )
        self.assertEqual(
            draft.lm_head.weight.data_ptr(), decoder.lm_head.weight.data_ptr()
        )
        results = self._generate(self.prompt_ids[:8], temperature=0.0, max_tokens=128)
        self._check_greedy(self.prompt_ids[:8], results)
