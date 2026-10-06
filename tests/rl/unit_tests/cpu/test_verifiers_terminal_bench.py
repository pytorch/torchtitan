# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.rl.controller import Controller
from torchtitan.rl.examples.verifiers.terminal_bench import taskset
from torchtitan.rl.examples.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from verifiers.v1.harnesses.terminus_2 import Terminus2Harness, Terminus2HarnessConfig
from verifiers.v1.serve import env_config_data
from verifiers.v1.tasksets.harbor import HarborEnvConfig
from verifiers.v1.utils.loaders import load_harness, resolve_env_config

TRAIN_DATASET = "local/tmax@v1"
EVAL_DATASET = "terminal-bench/terminal-bench-2-1"
MAX_CONTEXT_LENGTH = 32768
MAX_TURNS = 64
MAX_CONCURRENT_ROLLOUTS = 64


def _rollouter_config(train_dataset: str, validation_dataset: str):
    return terminal_bench_rollouter_config(
        train_dataset,
        validation_dataset,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_turns=MAX_TURNS,
        max_concurrent_rollouts=MAX_CONCURRENT_ROLLOUTS,
    )


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset() -> None:
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = config.verifiers_env_server.environment

    assert isinstance(environment, HarborEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, Terminus2HarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert config.generation_server.max_rollout_tokens == MAX_CONTEXT_LENGTH
    assert environment.agent.max_turns == MAX_TURNS
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.train_dataset.verifiers_taskset
    assert config.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    assert config.verifiers_env_server.local_taskset_module == taskset.__name__
    worker_config = resolve_env_config(env_config_data(environment))
    assert isinstance(load_harness(worker_config.agent.harness), Terminus2Harness)


def test_training_cannot_read_benchmark_as_training_data() -> None:
    with pytest.raises(ValueError, match="different datasets"):
        _rollouter_config(EVAL_DATASET, EVAL_DATASET)


def _terminal_bench_config(name: str) -> Controller.Config:
    return ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.rl.verifiers_terminal_bench",
            "--config",
            name,
        ]
    )


def test_training_recipe_uses_separate_datasets() -> None:
    config = _terminal_bench_config("rl_grpo_qwen35_9b_terminal_bench")
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.num_prompts_per_train_step == 12
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.target_offpolicy_steps == 3
    assert config.trainer.training.max_context_length == 131072
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.mixed_precision_param == "bfloat16"
    assert config.trainer.training.mixed_precision_reduce == "float32"
    (optimizer,) = config.trainer.optim.optimizer.optimizers
    assert optimizer.fused
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert isinstance(config.trainer.activation_checkpoint, FullAC.Config)
    assert config.trainer.checkpointer.interval == 20
    assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    assert config.num_generators == 8
    assert config.generator.parallelism.data_parallel_degree == 1
    assert config.rollouter.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.rollouter.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET


def _num_kv_heads(model: object) -> int:
    for layer in model.layers:
        attention = getattr(layer, "attention", None)
        if hasattr(attention, "n_kv_heads"):
            return attention.n_kv_heads
    raise AssertionError("model has no full-attention layer")


def _num_experts(model: object) -> int | None:
    for layer in model.layers:
        moe = getattr(layer, "moe", None)
        if moe is not None:
            return moe.num_experts
    return None


@pytest.mark.parametrize(
    ("name", "trainer_gpus", "num_generators", "gpus_per_generator"),
    [
        ("rl_grpo_qwen35_9b_terminal_bench", 8, 8, 1),
        ("rl_grpo_qwen35_35b_a3b_terminal_bench", 8, 2, 4),
    ],
)
def test_recipe_layouts_fit_the_model(
    name: str,
    trainer_gpus: int,
    num_generators: int,
    gpus_per_generator: int,
) -> None:
    """Each recipe's trainer and generator layouts respect the model's shape.

    The invariants are the ones a launch would otherwise trip over one at a
    time: tensor parallelism divides the KV heads in both roles, and for the MoE
    model expert parallelism divides the experts, is at least the trainer TP
    degree, and equals DP x TP in the generator. The GPU totals pin the intended
    16-GPU footprint.
    """
    config = _terminal_bench_config(name)
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism
    num_kv_heads = _num_kv_heads(config.model)
    num_experts = _num_experts(config.model)

    assert (
        trainer.data_parallel_replicate_degree
        * trainer.data_parallel_shard_degree
        * trainer.tensor_parallel_degree
        * trainer.context_parallel_degree
        == trainer_gpus
    )
    assert config.num_generators == num_generators
    assert generator.data_parallel_degree * generator.tensor_parallel_degree == (
        gpus_per_generator
    )
    assert num_kv_heads % trainer.tensor_parallel_degree == 0
    assert num_kv_heads % generator.tensor_parallel_degree == 0

    if num_experts is None:
        assert trainer.expert_parallel_degree == 1
        assert generator.expert_parallel_degree == 1
        assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    else:
        assert num_experts % trainer.expert_parallel_degree == 0
        assert trainer.expert_parallel_degree >= trainer.tensor_parallel_degree
        assert (
            trainer.data_parallel_shard_degree * trainer.tensor_parallel_degree
        ) % trainer.expert_parallel_degree == 0
        assert generator.expert_parallel_degree == (
            generator.data_parallel_degree * generator.tensor_parallel_degree
        )
        assert num_experts % generator.expert_parallel_degree == 0
        # The standard MoE dispatcher reads split sizes back to the host, which
        # CUDA graph capture does not allow.
        assert config.generator.cuda_graph.mode == "NONE"


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen35_9b_terminal_bench",
        "rl_grpo_qwen35_35b_a3b_terminal_bench",
    ],
)
def test_recipes_share_the_loop_and_keep_fp32_master_weights(name: str) -> None:
    """Every model trains on the same loop; only size-dependent settings differ.

    Master weights stay fp32 (the default): at a 1e-6 learning rate bf16
    parameters round most updates away. The env server must run every rollout
    the controller keeps in flight, or the excess queues and the generators idle.
    """
    config = _terminal_bench_config(name)
    assert config.trainer.training.dtype == "float32"
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert config.generator.sampling.max_tokens == 16384
    assert config.rollouter.verifiers_env_server.environment.agent.max_turns == 120
    assert (
        config.rollouter.generation_server.max_rollout_tokens
        == config.trainer.training.max_context_length
    )
    loop = config.async_loop
    serve = config.rollouter.verifiers_env_server.serve
    assert serve.pool.num_workers * serve.max_concurrent >= (
        loop.max_active_rollout_groups * loop.num_samples_per_prompt
    )
