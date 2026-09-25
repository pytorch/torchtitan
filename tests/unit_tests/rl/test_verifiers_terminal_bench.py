# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import ast
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.config.manager import ConfigManager
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.rl.controller import Controller
from torchtitan.rl.examples.verifiers import VerifiersTaskDataset
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.experiments.verifiers.terminal_bench import data
from torchtitan.rl.experiments.verifiers.terminal_bench.controller import (
    TerminalBenchController,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
    TerminalBenchTerminusHarnessConfig,
    terminus_program_source,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from verifiers.v1.errors import TaskError
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.serve import env_config_data
from verifiers.v1.tasksets.harbor import HarborData, HarborEnvConfig
from verifiers.v1.utils.loaders import load_harness, resolve_env_config


def _task_dir(root: Path, name: str, *, instruction: bool = True) -> None:
    task = root / "tasks" / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'schema_version = "1.1"\n[environment]\ndocker_image = "python:3.12-slim"\n'
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    if instruction:
        (task / "instruction.md").write_text("Solve it")


def test_local_taskset_is_ordered_and_requires_complete_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _task_dir(tmp_path, "b")
    _task_dir(tmp_path, "a")
    _task_dir(tmp_path, "incomplete", instruction=False)
    monkeypatch.setattr(
        data,
        "parse_task",
        lambda task_dir, idx, config: HarborData(
            idx=idx,
            name=task_dir.name,
            prompt="Solve it",
            image="python:3.12-slim",
            task_dir=str(task_dir),
        ),
    )

    config = data.TerminalTasksetConfig(
        id="local_terminal_tasks", tasks_root=tmp_path, expected_num_tasks=2
    )
    tasks = list(data.TerminalTaskset(config).load())
    assert [(task.data.idx, task.data.name) for task in tasks] == [
        (0, "a"),
        (1, "b"),
    ]
    with pytest.raises(ValueError, match="Expected 89 Harbor tasks"):
        list(
            data.TerminalTaskset(
                config.model_copy(update={"expected_num_tasks": 89})
            ).load()
        )


def test_local_taskset_rejects_missing_root(tmp_path: Path) -> None:
    config = data.TerminalTasksetConfig(
        id="local_terminal_tasks", tasks_root=tmp_path / "absent"
    )
    with pytest.raises(ValueError, match="not a directory"):
        list(data.TerminalTaskset(config).load())


def test_harbor_task_data_reaches_verifiers_dataset(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    _task_dir(tmp_path, "check-fixtures")
    assert data.TerminalTaskset.task_type() is data.TerminalTask
    alias = register_local_taskset_alias(data.__name__)
    dataset = VerifiersTaskDataset.Config(
        verifiers_taskset=data.TerminalTasksetConfig(
            id=alias, tasks_root=tmp_path, expected_num_tasks=1
        ),
        shuffle=False,
    ).build()
    sample = next(dataset).verifiers_task_data
    assert sample["prompt"] == "Solve it"
    assert sample["image"] == "python:3.12-slim"
    assert sample["task_dir"] == str(tmp_path / "tasks" / "check-fixtures")


def test_image_overrides_keep_the_task_tree_unchanged(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    _task_dir(tmp_path, "override")
    image_map = tmp_path / "images.json"
    image_map.write_text(json.dumps({"override": "registry.example/with-tmux:v1"}))
    config = data.TerminalTasksetConfig(
        id="terminal_tasks", tasks_root=tmp_path, image_overrides_path=image_map
    )
    task = next(data.TerminalTaskset(config).load())
    assert task.data.image == "registry.example/with-tmux:v1"
    assert (
        'docker_image = "python:3.12-slim"'
        in (tmp_path / "tasks" / "override" / "task.toml").read_text()
    )
    task_dir = tmp_path / "tasks" / "override"
    (task_dir / "task.toml").write_text('schema_version = "1.1"\n')
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    assert next(data.TerminalTaskset(config).load()).data.image == (
        "registry.example/with-tmux:v1"
    )
    with pytest.raises(ValueError, match="Dockerfile"):
        list(
            data.TerminalTaskset(
                config.model_copy(update={"image_overrides_path": None})
            ).load()
        )
    image_map.write_text("{}")
    with pytest.raises(ValueError, match="No image override"):
        list(data.TerminalTaskset(config).load())


def test_missing_tmux_reports_a_preparation_error(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    _task_dir(tmp_path, "missing-tmux")
    task = next(
        data.TerminalTaskset(
            data.TerminalTasksetConfig(id="terminal_tasks", tasks_root=tmp_path)
        ).load()
    )

    class RuntimeWithoutTmux:
        async def run(self, argv, env):
            return SimpleNamespace(exit_code=1)

    with pytest.raises(TaskError, match="lacks tmux"):
        asyncio.run(task.setup(RuntimeWithoutTmux()))


@pytest.mark.skipif(
    os.environ.get("TERMINAL_BENCH_DOCKER_SMOKE") != "1",
    reason="requires Docker and a pullable test image",
)
def test_harbor_verifier_grades_in_docker(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    _task_dir(tmp_path, "docker-grading")
    task = next(
        data.TerminalTaskset(
            data.TerminalTasksetConfig(id="terminal_tasks", tasks_root=tmp_path)
        ).load()
    )
    assert task.data.image is not None

    async def grade() -> float | dict[str, float]:
        async with provision_runtime(
            vf.DockerConfig(image=task.data.image, workdir="/"),
            env=task.runtime_env(),
        ) as runtime:
            await runtime.prepare_setup()
            await runtime.prepare_execution([])
            return await task.solved(
                runtime, SimpleNamespace(record_metrics=lambda metrics: None)
            )

    assert asyncio.run(grade()) == 1.0


def test_published_image_workdir_is_preserved(tmp_path: Path) -> None:
    dockerfile = tmp_path / "environment" / "Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM python:3.12\nWORKDIR /app\nWORKDIR /app/project\n")
    assert data._dockerfile_workdir(tmp_path) == "/app/project"


def test_published_terminal_bench_tree_if_available(tmp_path: Path) -> None:
    root = os.environ.get("TERMINAL_BENCH_PUBLIC_TASKS_ROOT")
    if not root:
        pytest.skip("set TERMINAL_BENCH_PUBLIC_TASKS_ROOT to a pinned 2.1 task tree")

    task_dirs = [path.parent for path in Path(root).rglob("task.toml")]
    assert len(task_dirs) == 89
    image_map = tmp_path / "images.json"
    image_map.write_text(
        json.dumps({task_dir.name: "python:3.12-slim" for task_dir in task_dirs})
    )
    tasks = list(
        data.TerminalTaskset(
            data.TerminalTasksetConfig(
                id="local_terminal_tasks",
                tasks_root=Path(root),
                image_overrides_path=image_map,
                expected_num_tasks=89,
            )
        ).load()
    )
    assert len(tasks) == 89
    assert len({task.data.name for task in tasks}) == 89


def test_terminus_program_keeps_coworker_xml_scaffold() -> None:
    source = terminus_program_source().replace("{version}", "0.22.0")
    ast.parse(source)
    assert source.count('parser_name="xml"') == 1
    assert source.count("enable_summarize=False") == 1
    assert source.count("max_turns=120") == 1


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset(
    tmp_path: Path,
) -> None:
    config = terminal_bench_rollouter_config(
        tmp_path / "train", tmp_path / "terminal-bench-2.1"
    )
    environment = config.verifiers_env_server.environment

    assert isinstance(environment, HarborEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert environment.agent.max_turns == 120
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.train_dataset.verifiers_taskset
    assert config.validation_dataset.verifiers_taskset.expected_num_tasks == 89
    assert config.verifiers_env_server.local_taskset_module == data.__name__
    worker_config = resolve_env_config(env_config_data(environment))
    assert isinstance(worker_config.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert isinstance(
        load_harness(worker_config.agent.harness), TerminalBenchTerminusHarness
    )


def test_training_cannot_read_benchmark_as_training_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different trees"):
        terminal_bench_rollouter_config(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="different trees"):
        terminal_bench_rollouter_config(tmp_path, tmp_path / "eval")
    assert (
        terminal_bench_rollouter_config(
            tmp_path, tmp_path, eval_only=True
        ).train_dataset.verifiers_taskset.expected_num_tasks
        == 89
    )


def test_config_registration_and_eval_only_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_BENCH_EVAL_TASKS_ROOT", str(tmp_path / "eval"))
    monkeypatch.setenv("TERMINAL_BENCH_CHECKPOINT", str(tmp_path / "checkpoint"))
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan.rl.experiments.verifiers.terminal_bench",
            "--config",
            "rl_grpo_qwen35_9b_terminal_bench_eval",
        ]
    )
    assert config.model.dim == 4096
    assert config.model.max_context_length == 65536
    assert config.async_loop.num_training_steps == 0
    assert config.eval_only
    assert config.async_loop.validation.num_samples == 89
    assert config.trainer.checkpointer.initial_load_path == str(tmp_path / "checkpoint")
    assert not config.trainer.checkpointer.initial_load_in_hf


def test_eval_only_runs_one_validation_without_training() -> None:
    controller = object.__new__(TerminalBenchController)
    controller.config = SimpleNamespace(eval_only=True)
    controller.start_step = 0
    controller._validate_and_log = AsyncMock()

    asyncio.run(controller.run())

    controller._validate_and_log.assert_awaited_once_with(step=0)


def test_training_uses_standard_titanrl_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_loop = AsyncMock()
    monkeypatch.setattr(Controller, "run", main_loop)
    controller = object.__new__(TerminalBenchController)
    controller.config = SimpleNamespace(eval_only=False)

    asyncio.run(controller.run())

    main_loop.assert_awaited_once_with()


def test_training_recipe_uses_separate_frozen_task_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERMINAL_BENCH_TRAIN_TASKS_ROOT", str(tmp_path / "train"))
    monkeypatch.setenv("TERMINAL_BENCH_EVAL_TASKS_ROOT", str(tmp_path / "eval"))
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan.rl.experiments.verifiers.terminal_bench",
            "--config",
            "rl_grpo_qwen35_9b_terminal_bench",
        ]
    )
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.trainer.training.max_context_length == 65536
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.mixed_precision_param == "bfloat16"
    assert config.trainer.training.mixed_precision_reduce == "float32"
    assert config.trainer.optimizer.implementation == "fused"
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert isinstance(config.trainer.activation_checkpoint, FullAC.Config)
    assert config.trainer.checkpointer.interval == 20
    assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    assert config.num_generators == 8
    assert config.generator.parallelism.data_parallel_degree == 1
    assert config.rollouter.train_dataset.verifiers_taskset.tasks_root == (
        tmp_path / "train"
    )
    assert config.rollouter.validation_dataset.verifiers_taskset.tasks_root == (
        tmp_path / "eval"
    )
