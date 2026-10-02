# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import ast
import json
import subprocess
import sys

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.rl.controller import Controller
from torchtitan.rl.examples.verifiers.terminal_bench import taskset
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
    TerminalBenchTerminusHarnessConfig,
    terminus_program_source,
)
from torchtitan.rl.examples.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from verifiers.v1.serve import env_config_data
from verifiers.v1.tasksets.harbor import HarborEnvConfig
from verifiers.v1.utils.loaders import load_harness, resolve_env_config

TRAIN_DATASET = "local/tmax@v1"
EVAL_DATASET = "terminal-bench/terminal-bench-2-1"
MAX_CONTEXT_LENGTH = 32768
MAX_TOKENS = 4096


def _rollouter_config(train_dataset: str, validation_dataset: str):
    return terminal_bench_rollouter_config(
        train_dataset,
        validation_dataset,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_tokens=MAX_TOKENS,
    )


def test_terminus_program_keeps_coworker_xml_scaffold() -> None:
    harness = _rollouter_config(
        TRAIN_DATASET, EVAL_DATASET
    ).verifiers_env_server.environment.agent.harness
    source = terminus_program_source(harness)
    ast.parse(source)
    assert source.count('parser_name="xml"') == 1
    assert source.count("enable_summarize=False") == 1
    assert source.count("max_turns=120") == 1
    assert source.count('"harbor==0.22.0"') == 1
    assert (
        source.count(
            f"model_info={{'max_input_tokens': {MAX_CONTEXT_LENGTH}, "
            f"'max_output_tokens': {MAX_TOKENS}}}"
        )
        == 1
    )


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset() -> None:
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = config.verifiers_env_server.environment

    assert isinstance(environment, HarborEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert environment.agent.harness.max_input_tokens == MAX_CONTEXT_LENGTH
    assert environment.agent.harness.max_output_tokens == MAX_TOKENS
    assert config.generation_server.max_rollout_tokens == MAX_CONTEXT_LENGTH
    assert environment.agent.max_turns == 120
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.train_dataset.verifiers_taskset
    assert config.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    assert config.verifiers_env_server.local_taskset_module == taskset.__name__
    worker_config = resolve_env_config(env_config_data(environment))
    assert isinstance(worker_config.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert isinstance(
        load_harness(worker_config.agent.harness), TerminalBenchTerminusHarness
    )


def test_worker_process_resolves_the_harness_from_a_fresh_interpreter() -> None:
    """The env-server worker shares no ``sys.modules`` with the controller.

    It imports only the local taskset module and then rebuilds the environment
    config from JSON, so that one import must be enough to make the harness id
    resolvable. Resolving in the test process would pass regardless, because the
    controller side has already registered the alias there.
    """
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = json.dumps(env_config_data(config.verifiers_env_server.environment))
    worker = f"""
import json
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from verifiers.v1.utils.loaders import load_harness, resolve_env_config

environment = json.loads({environment!r})
environment["taskset"]["id"] = register_local_taskset_alias(
    {config.verifiers_env_server.local_taskset_module!r}
)
env_config = resolve_env_config(environment)
print(type(load_harness(env_config.agent.harness)).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", worker], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("TerminalBenchTerminusHarness")


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
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.trainer.training.max_context_length == 65536
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
    parameters round most updates away.
    """
    config = _terminal_bench_config(name)
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.max_context_length == 65536
    assert config.async_loop.num_prompts_per_train_step == 8
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert config.generator.sampling.max_tokens == 16384
    harness = config.rollouter.verifiers_env_server.environment.agent.harness
    assert harness.max_input_tokens == config.trainer.training.max_context_length
    assert harness.max_output_tokens == config.generator.sampling.max_tokens
    assert (
        config.rollouter.generation_server.max_rollout_tokens
        == config.trainer.training.max_context_length
    )
