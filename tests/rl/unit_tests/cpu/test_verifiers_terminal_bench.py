# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import asyncio
import itertools
import sys
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from renderers import CustomRendererConfig

from torchtitan.components.renderer import RenderersConfigAdapter
from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.examples.verifiers.terminal_bench import taskset
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan_recipes.rl.verifiers_terminal_bench import (
    _terminal_bench_rollouter_config,
)
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.tasksets.harbor import taskset as harbor_taskset
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.loaders import (
    load_environment,
    load_harness,
    resolve_env_config,
)

TRAIN_DATASET = "local/tmax@v1"
EVAL_DATASET = "terminal-bench/terminal-bench-2-1"
MAX_CONTEXT_LENGTH = 32768
MAX_TURNS = 64


def _rollouter_config(train_dataset: str, validation_dataset: str):
    return _terminal_bench_rollouter_config(
        train_dataset,
        validation_dataset,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_turns=MAX_TURNS,
    )


class _RecordingRuntime:
    """Records the program the harness prepares and every command it runs, in order."""

    def __init__(self) -> None:
        # The methods below stand in for Verifiers' Runtime; fail if it renames one.
        assert all(
            hasattr(Runtime, name) for name in ("prepare_uv_script", "run", "write")
        )
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.files: dict[str, bytes] = {}

    async def prepare_uv_script(self, source: str, env: dict[str, str]) -> list[str]:
        self.source = source
        return ["terminus_harness"]

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data


def _run_terminus_program(
    tmp_path, monkeypatch, system_prompt: str | None = None, **options: bool
) -> SimpleNamespace:
    """Launch the harness with ``options``, then run the program it prepared with a
    recording stand-in for Harbor's ``Terminus2``.

    Returns the program's argv, the keyword arguments it passes to ``Terminus2``
    (``kwargs``), the agent it built (``terminus``), and the calls it makes on the
    agent, in order (``calls``).
    """
    harbor_terminus = pytest.importorskip("harbor.agents.terminus_2")
    from harbor.models.trial.paths import EnvironmentPaths

    harness_config = _rollouter_config(
        TRAIN_DATASET, EVAL_DATASET
    ).worker.environment.agent.harness
    harness = load_harness(harness_config.model_copy(update=options))
    runtime = _RecordingRuntime()
    asyncio.run(
        harness.launch(
            SimpleNamespace(model="torchtitan"),
            SimpleNamespace(id="trace"),
            runtime,
            "http://127.0.0.1:1/v1",
            "secret",
            {},
            SimpleNamespace(
                prompt="Fix the parser in /app.", system_prompt=system_prompt
            ),
        )
    )
    assert runtime.source.count('"harbor==0.22.0"') == 1
    # Run the program on a local copy of the prompts file the harness wrote.
    [(prompts_path, prompts)] = runtime.files.items()
    local_prompts = tmp_path / "prompts.json"
    local_prompts.write_bytes(prompts)
    [(program_argv, _)] = runtime.commands
    argv = [arg.replace(prompts_path, str(local_prompts)) for arg in program_argv]

    recorded = SimpleNamespace(argv=program_argv, kwargs={}, calls=[])

    async def llm_call(*args: object, message_history: list[dict]) -> list[dict]:
        return message_history

    class RecordingSession:
        async def send_keys(
            self, keys: list[str], min_timeout_sec: float = 0.0
        ) -> None:
            recorded.calls.append(("send_keys", keys, min_timeout_sec))

    class RecordingTerminus2:
        def __init__(self, **kwargs: object) -> None:
            recorded.kwargs.update(kwargs)
            recorded.terminus = self
            self._llm = SimpleNamespace(call=llm_call)
            self._session = RecordingSession()

        async def setup(self, environment: object) -> None:
            recorded.calls.append(("setup",))

        async def run(self, instruction: str, *args: object) -> None:
            recorded.calls.append(("run", instruction))

    monkeypatch.setattr(harbor_terminus, "Terminus2", RecordingTerminus2)
    monkeypatch.setattr(EnvironmentPaths, "agent_dir", PurePosixPath("/"))
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path / "tmux"))
    monkeypatch.setattr(sys, "argv", argv)
    program: dict[str, object] = {"__name__": "terminus_harness"}
    exec(compile(runtime.source, "terminus_harness.py", "exec"), program)
    asyncio.run(program["main"]())
    return recorded


def test_harness_options_reach_terminus2(tmp_path, monkeypatch) -> None:
    defaults = _run_terminus_program(tmp_path, monkeypatch).kwargs
    assert defaults["interleaved_thinking"] is True
    assert defaults["enable_summarize"] is False

    flipped = _run_terminus_program(
        tmp_path, monkeypatch, interleaved_thinking=False, enable_summarize=True
    ).kwargs
    assert flipped["interleaved_thinking"] is False
    assert flipped["enable_summarize"] is True


def test_prompts_reach_terminus2_outside_its_argv(tmp_path, monkeypatch) -> None:
    """The program runs next to the agent's commands, so an agent's ``pkill -f
    parser`` must not match it."""
    recorded = _run_terminus_program(
        tmp_path, monkeypatch, system_prompt="Answer in JSON."
    )
    assert not any("parser" in arg or "JSON" in arg for arg in recorded.argv)
    assert recorded.calls[-1] == ("run", "Fix the parser in /app.")
    history = asyncio.run(
        recorded.terminus._llm.call(message_history=[{"role": "user", "content": "hi"}])
    )
    assert history[0] == {"role": "system", "content": "Answer in JSON."}


def test_agent_shell_runs_inside_the_login_shell(tmp_path, monkeypatch) -> None:
    """After setup, the program types ``bash`` and ``clear``, as Harbor's terminal recording does."""
    assert _run_terminus_program(tmp_path, monkeypatch).calls == [
        ("setup",),
        ("send_keys", ["bash", "Enter"], 1.0),
        ("send_keys", ["clear", "Enter"], 0.0),
        ("run", "Fix the parser in /app."),
    ]


def test_tmux_server_outlives_launch_until_cleanup() -> None:
    """Verifiers scores between ``launch`` and ``cleanup``; the agent's tmux session
    and the jobs started from it must still be alive then."""
    harness = TerminalBenchTerminusHarness(TerminalBenchTerminusHarnessConfig())
    runtime = _RecordingRuntime()
    trace = SimpleNamespace(id="trace")
    asyncio.run(
        harness.launch(
            SimpleNamespace(model="torchtitan"),
            trace,
            runtime,
            "http://127.0.0.1:1/v1",
            "secret",
            {},
            SimpleNamespace(prompt="Fix the parser in /app.", system_prompt=None),
        )
    )
    # Launch ran the program only.
    [(_, program_env)] = runtime.commands

    asyncio.run(harness.cleanup(trace, runtime))
    [_, (argv, env)] = runtime.commands
    assert "tmux kill-server" in argv[-1]
    assert env == {"TMUX_TMPDIR": "/tmp/vf-terminus-2-trace"}
    assert program_env["TMUX_TMPDIR"] == env["TMUX_TMPDIR"]
    # The prompts file cleanup removes is the one launch wrote.
    assert '"$TMUX_TMPDIR.prompts.json"' in argv[-1]
    assert list(runtime.files) == ["/tmp/vf-terminus-2-trace.prompts.json"]


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset() -> None:
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = config.worker.environment

    assert isinstance(environment, vf.SingleAgentEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert config.worker.generation_server.max_rollout_tokens == MAX_CONTEXT_LENGTH
    assert environment.agent.max_turns == MAX_TURNS
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.training_dataloader.dataset.verifiers_taskset
    assert config.training_dataloader.dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    # What each rollout worker loads.
    env = load_environment(resolve_env_config(environment))
    assert type(env.taskset) is taskset.TerminalTaskset
    assert isinstance(
        load_harness(env.config.agent.harness), TerminalBenchTerminusHarness
    )


def _write_harbor_task(task_dir, *, task_toml: str, dockerfile: str | None) -> None:
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    (task_dir / "instruction.md").write_text("Fix the parser.\n")
    (task_dir / "task.toml").write_text(task_toml)
    if dockerfile is not None:
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text(dockerfile)


def test_tasks_run_in_the_image_workdir_and_grade_with_task_timeouts(
    tmp_path, monkeypatch
) -> None:
    """Each task gets the start folder and grading timeout that `harbor run` gives it.

    A Terminal-Bench 2.1 task names its folder only in its Dockerfile and declares a
    verifier timeout. A TMax task from prepare_tmax.py names its folder in task.toml
    and declares no verifier timeout, so it gets Harbor's 600 s. A WORKDIR that Docker
    would expand ($HOME) is left to the runtime's default folder.
    """
    monkeypatch.setattr(harbor_taskset, "CACHE", tmp_path)
    for dataset_dir in ("local_train_v1", "local_validation_v1"):
        _write_harbor_task(
            tmp_path / dataset_dir / "terminal-bench",
            task_toml='[task]\nname = "test/terminal-bench"\n'
            '[environment]\ndocker_image = "example/dclm:1"\n'
            "[verifier]\ntimeout_sec = 900.0\n",
            dockerfile="FROM python:3.13\nWORKDIR /app\nRUN true\nWORKDIR /app/dclm\n",
        )
        _write_harbor_task(
            tmp_path / dataset_dir / "tmax",
            task_toml='[task]\nname = "test/tmax"\n'
            '[environment]\ndocker_image = "example/tmax:1"\nworkdir = "/home/user"\n',
            dockerfile=None,
        )
        _write_harbor_task(
            tmp_path / dataset_dir / "env-var",
            task_toml='[task]\nname = "test/env-var"\n'
            '[environment]\ndocker_image = "example/env:1"\n',
            dockerfile="FROM python:3.13\nWORKDIR $HOME/work\n",
        )
    config = _rollouter_config("local/train@v1", "local/validation@v1")
    agent = config.worker.environment.agent
    task_type = taskset.TerminalTaskset.task_type()

    for dataset in (config.training_dataloader.dataset, config.validation_dataset):
        runs = {}
        # A rollout worker rebuilds each task from the data the controller sends.
        for sample in itertools.islice(dataset.build(), 3):
            task = task_type(
                task_type.data_type().model_validate(sample.verifiers_task_data)
            )
            runs[task.data.name] = (
                resolve_runtime_config(agent.runtime, task).workdir,
                resolve_rollout_timeouts(agent.timeout, task).scoring,
            )
        assert runs == {
            "test/terminal-bench": ("/app/dclm", 900.0),
            "test/tmax": ("/home/user", 600.0),
            "test/env-var": (None, 600.0),
        }
    # With Verifiers' default ignore_timeouts=True, no task gets a grading timeout.
    ignoring = config.validation_dataset.verifiers_taskset.model_copy(
        update={"ignore_timeouts": True}
    )
    scorings = [task.data.timeout.scoring for task in taskset.TerminalTaskset(ignoring)]
    assert scorings == [None, None, None]


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
    assert config.trainer.model.max_context_length == 131072
    assert config.precision.dtype == "float32"
    assert config.precision.param_dtype == "bfloat16"
    assert config.precision.reduce_dtype == "float32"
    (optimizer,) = config.trainer.optim.optimizer.optimizers
    assert optimizer.fused
    assert config.async_loop.batcher.training_sample_builder.drop_zero_std_reward_groups
    assert isinstance(config.trainer.activation_remat, FullActivationRematConfig)
    assert config.trainer.checkpointer.interval == 20
    assert config.generator.engine.cuda_graph.mode == "FULL_DECODE_ONLY"
    assert config.num_generators == 8
    assert config.generator.parallelism.data_parallel_degree == 1
    assert (
        config.rollouter.training_dataloader.dataset.verifiers_taskset.dataset
        == TRAIN_DATASET
    )
    assert config.rollouter.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    # `Controller.__init__` dumps the config. The Verifiers worker's own `to_dict` turns its
    # pydantic env config into a dict; the generic dump would give its repr, a string.
    assert isinstance(config.to_dict()["rollouter"]["worker"]["environment"], dict)


def _num_kv_heads(model: object) -> int:
    for layer in model.layers:
        attention = getattr(layer, "attention", None)
        if hasattr(attention, "n_kv_heads"):
            return attention.n_kv_heads
    raise AssertionError("model has no full-attention layer")


def _num_experts(model: object) -> int | None:
    moe = model.first_moe
    return moe.num_experts if moe is not None else None


def test_muse_glimmer_terminal_bench_recipe_uses_worker_loadable_renderer() -> None:
    config = _terminal_bench_config("rl_grpo_muse_glimmer_30b_terminal_bench")

    assert config.hf_assets_path == "./assets/hf/Muse-Glimmer-30B"
    assert isinstance(config.renderer, RenderersConfigAdapter)
    worker_config = config.renderer.get_renderers_config()
    assert isinstance(worker_config, CustomRendererConfig)
    assert worker_config.import_path == (
        "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
    )
    assert worker_config.model_extra is not None
    assert worker_config.model_extra["reasoning_strength"] == "low"

    assert config.trainer.model.max_context_length == 16384
    assert config.generator.model.max_context_length == 16384
    assert config.rollouter.worker.generation_server.max_rollout_tokens == 16384
    assert config.generator.sampling.max_tokens == 4096
    assert config.rollouter.worker.environment.agent.max_turns == 8
    assert config.generator.engine.cuda_graph.mode == "NONE"
    assert isinstance(config.trainer.activation_remat, FullActivationRematConfig)
    assert _num_kv_heads(config.generator.model) == 2
    assert config.trainer.parallelism.tensor_parallel_degree == 2
    assert config.generator.parallelism.tensor_parallel_degree == 2
    assert config.trainer.parallelism.data_parallel_shard_degree == 4
    assert config.num_generators == 4
    assert (
        config.rollouter.training_dataloader.dataset.verifiers_taskset.dataset
        == TRAIN_DATASET
    )
    assert config.rollouter.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET


@pytest.mark.parametrize(
    ("name", "trainer_gpus", "num_generators", "gpus_per_generator"),
    [
        ("rl_grpo_qwen35_9b_terminal_bench", 8, 8, 1),
        ("rl_dapo_qwen35_35b_a3b_terminal_bench_dist_moe", 4, 16, 1),
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
    degree, and equals DP x TP in the generator. The GPU totals pin each recipe's
    footprint.
    """
    config = _terminal_bench_config(name)
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism
    num_kv_heads = _num_kv_heads(config.generator.model)
    num_experts = _num_experts(config.generator.model)

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
        assert config.generator.engine.cuda_graph.mode == "FULL_DECODE_ONLY"
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


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen35_9b_terminal_bench",
        "rl_dapo_qwen35_35b_a3b_terminal_bench_dist_moe",
    ],
)
def test_recipes_share_the_loop_and_keep_fp32_master_weights(name: str) -> None:
    """Every model trains on the same loop; only size-dependent settings differ.

    Master weights stay fp32 (the default): at a 1e-6 learning rate bf16
    parameters round most updates away.
    """
    config = _terminal_bench_config(name)
    assert config.precision.dtype == "float32"
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.batcher.training_sample_builder.drop_zero_std_reward_groups
    assert config.generator.sampling.max_tokens == 16384
    agent = config.rollouter.worker.environment.agent
    assert agent.max_turns == 120
    assert (
        config.rollouter.worker.generation_server.max_rollout_tokens
        == config.trainer.model.max_context_length
    )
