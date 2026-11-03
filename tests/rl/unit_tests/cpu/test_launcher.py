# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import shlex
import sys
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl.actors.monarch import system as monarch_system
from torchtitan.rl.actors.monarch.launcher import submit
from torchtitan.rl.actors.monarch.launcher.base import (
    LaunchPlan,
    Layout,
    LAYOUT_ENV,
    MonarchControllerWithLauncher,
    MonarchLauncher,
)
from torchtitan.rl.actors.monarch.launcher.slurm import SlurmLauncher
from torchtitan.rl.actors.monarch.provisioning import _check_gpus_per_host, HostMeshes
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig


class FakeLauncher(MonarchLauncher):
    @dataclass(kw_only=True, slots=True)
    class Config(MonarchLauncher.Config):
        events: list = field(default_factory=list)

    def gpus_per_node(self) -> int:
        return 8

    def submit(self, plan: LaunchPlan) -> str:
        self.config.events.append("submit")
        return "job-123"

    async def acquire_hosts(self, layout: Layout) -> dict:
        self.config.events.append("acquire_hosts")
        return {name: f"hosts[{name}]" for name in layout.mesh_nodes}


def _rl_config(*, trainer_dp=4, generator_tp=2, num_generators=1, **kwargs):
    return SimpleNamespace(
        trainer=SimpleNamespace(
            parallelism=ParallelismConfig(data_parallel_shard_degree=trainer_dp)
        ),
        generator=SimpleNamespace(
            parallelism=InferenceParallelismConfig(tensor_parallel_degree=generator_tp)
        ),
        num_generators=num_generators,
        resolve_overrides=lambda: None,
        **kwargs,
    )


def _plan(launcher_config, **kwargs):
    config = _rl_config(dump_folder="/out", **kwargs)
    return launcher_config.build().plan(config, ["--module", "m"])


# FakeLauncher has 8 GPUs per node. Sizes are (trainer, generator, num_generators).
_LAYOUT_CASES = {
    "one_gpu_generators_share_a_node": (
        (8, 1, 8),
        {"trainer": 1, "node_0": 1},
        {
            "trainer": "trainer",
            **{f"generator_{i}": "node_0" for i in range(8)},
        },
        0,
    ),
    "trainer_shares_its_node": (
        (4, 4, 3),
        {"node_0": 1, "node_1": 1},
        {
            "trainer": "node_0",
            "generator_0": "node_0",
            "generator_1": "node_1",
            "generator_2": "node_1",
        },
        0,
    ),
    "multi_node_roles": (
        (16, 8, 2),
        {"trainer": 2, "generator_0": 1, "generator_1": 1},
        {
            "trainer": "trainer",
            "generator_0": "generator_0",
            "generator_1": "generator_1",
        },
        0,
    ),
    "generator_does_not_fit_trainer_node": (
        (6, 4, 2),
        {"node_0": 1, "node_1": 1},
        {"trainer": "node_0", "generator_0": "node_1", "generator_1": "node_1"},
        2,
    ),
    "whole_node_generator_after_small_trainer": (
        (4, 8, 1),
        {"node_0": 1, "generator": 1},
        {"trainer": "node_0", "generator": "generator"},
        4,
    ),
}


@pytest.mark.parametrize(
    "sizes, mesh_nodes, role_mesh, idle_gpus",
    _LAYOUT_CASES.values(),
    ids=_LAYOUT_CASES.keys(),
)
def test_layout(sizes, mesh_nodes, role_mesh, idle_gpus, caplog):
    trainer_dp, generator_tp, num_generators = sizes
    layout = _plan(
        FakeLauncher.Config(),
        trainer_dp=trainer_dp,
        generator_tp=generator_tp,
        num_generators=num_generators,
    ).layout

    assert layout.mesh_nodes == mesh_nodes
    assert layout.role_mesh == role_mesh
    idle_warnings = [r.getMessage() for r in caplog.records if "idle" in r.getMessage()]
    if idle_gpus:
        assert len(idle_warnings) == 1
        assert idle_warnings[0].startswith(f"{idle_gpus} of ")
    else:
        assert idle_warnings == []


def test_layout_rejects_partial_multi_node_role():
    with pytest.raises(ValueError, match=r"trainer world size \(12\)"):
        _plan(FakeLauncher.Config(), trainer_dp=12, generator_tp=8)


@pytest.mark.parametrize(
    "sizes", [case[0] for case in _LAYOUT_CASES.values()], ids=_LAYOUT_CASES.keys()
)
def test_layout_fits_the_hosts(sizes):
    trainer_dp, generator_tp, num_generators = sizes
    layout = _plan(
        FakeLauncher.Config(),
        trainer_dp=trainer_dp,
        generator_tp=generator_tp,
        num_generators=num_generators,
    ).layout
    meshes = {
        name: [object()] * num_nodes for name, num_nodes in layout.mesh_nodes.items()
    }

    # Raises if the roles sharing a mesh need more GPUs than its hosts have.
    _check_gpus_per_host(layout.host_meshes(meshes), trainer_dp, generator_tp)


def test_layout_host_meshes_shares_one_object():
    layout = Layout(
        gpus_per_node=8,
        mesh_nodes={"node_0": 1, "node_1": 1},
        role_mesh={
            "trainer": "node_0",
            "generator_0": "node_0",
            "generator_1": "node_1",
            "generator_2": "node_1",
        },
    )
    node_0, node_1 = object(), object()

    host_meshes = layout.host_meshes({"node_0": node_0, "node_1": node_1})

    assert host_meshes.trainer is node_0
    assert host_meshes.generators[0] is node_0
    assert host_meshes.generators[1] is node_1
    assert host_meshes.generators[2] is node_1


def test_layout_host_meshes_maps_generators_by_name():
    layout = Layout(
        gpus_per_node=8,
        mesh_nodes={"trainer": 1, "generator_1": 1, "generator_0": 1},
        role_mesh={
            "generator_1": "generator_1",
            "trainer": "trainer",
            "generator_0": "generator_0",
        },
    )

    host_meshes = layout.host_meshes({name: name for name in layout.mesh_nodes})

    assert host_meshes.trainer == "trainer"
    assert host_meshes.generators == ["generator_0", "generator_1"]


def test_plan_sets_env_and_names_job():
    plan = _plan(FakeLauncher.Config(job_name="my-job", env={"A": "1"}))

    assert plan.job_name.startswith("my-job-") and len(plan.job_name) == 13
    assert plan.env == {"A": "1", LAYOUT_ENV: plan.layout.to_json()}
    assert plan.controller_args == ["--module", "m"]


def _run_submit(monkeypatch, config, argv):
    monkeypatch.setattr(submit.ConfigLoader, "load", lambda self, args: config)
    # The fake config stands in for a MonarchControllerWithLauncher.Config.
    monkeypatch.setattr(
        submit,
        "MonarchControllerWithLauncher",
        SimpleNamespace(Config=SimpleNamespace),
    )
    return submit.main(argv)


def _with_launcher(tmp_path, launcher):
    return _rl_config(dump_folder=str(tmp_path), launcher=launcher)


def test_submit_dry_run_does_not_submit(monkeypatch, tmp_path, capsys):
    events = []
    config = _with_launcher(tmp_path, FakeLauncher.Config(events=events))

    assert _run_submit(monkeypatch, config, ["--dry-run", "--module", "m"]) is None
    assert events == []
    assert (
        "mesh node_0: 1 node(s), roles trainer, generator\n" in capsys.readouterr().out
    )


def test_submit_plans_after_overrides(monkeypatch, tmp_path, capsys):
    config = _with_launcher(tmp_path, FakeLauncher.Config())

    def resolve_overrides():
        # Stands in for an override that adds a generator.
        config.num_generators = 2

    config.resolve_overrides = resolve_overrides

    _run_submit(monkeypatch, config, ["--dry-run", "--module", "m"])

    out = capsys.readouterr().out
    assert "mesh node_0: 1 node(s), roles trainer, generator_0, generator_1" in out


def test_submit_submits_and_returns_the_job_handle(monkeypatch, tmp_path):
    events = []
    config = _with_launcher(tmp_path, FakeLauncher.Config(events=events))

    assert _run_submit(monkeypatch, config, ["--module", "m"]) == "job-123"
    assert events == ["submit"]


def test_submit_requires_a_launch_config(monkeypatch, tmp_path):
    config = _with_launcher(tmp_path, FakeLauncher.Config())
    monkeypatch.setattr(submit.ConfigLoader, "load", lambda self, args: config)

    with pytest.raises(
        ValueError, match="Expected a MonarchControllerWithLauncher.Config"
    ):
        submit.main(["--module", "m"])


def test_launch_config_builds_the_launch_controller():
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_flex

    launch_config = MonarchControllerWithLauncher.Config.from_controller_config(
        rl_grpo_qwen3_0_6b_flex(), launcher=FakeLauncher.Config()
    )

    # A plain Controller would build the default actor system on the job's head
    # node only, without the launcher's hosts.
    assert launch_config._owner is MonarchControllerWithLauncher


def test_launch_config_requires_a_monarch_actor_system():
    from torchtitan.rl.actors.system import ActorSystem
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_flex

    config = rl_grpo_qwen3_0_6b_flex()
    config.actor_system = ActorSystem.Config()

    with pytest.raises(ValueError, match="MonarchActorSystem.Config"):
        MonarchControllerWithLauncher.Config.from_controller_config(
            config, launcher=FakeLauncher.Config()
        )


async def _stop_ok():
    pass


async def _stop_fails():
    raise RuntimeError("stop failed")


@pytest.fixture
def spawned(monkeypatch):
    """Stub proc-mesh spawning; record the host meshes `provision` passes."""
    spawned = {}

    def spawn_proc_mesh(trainer_ws, generator_ws, **kwargs):
        spawned["host_meshes"] = kwargs["host_meshes"]
        return SimpleNamespace(stop=_stop_ok), []

    async def setup_torch_elastic_env(mesh):
        pass

    monkeypatch.setattr(monarch_system, "spawn_proc_mesh", spawn_proc_mesh)
    monkeypatch.setattr(
        monarch_system,
        "spawn_cpu_procs",
        lambda n, **kwargs: SimpleNamespace(stop=_stop_fails),
    )
    monkeypatch.setattr(monarch_system, "batcher_host_mesh", lambda host_meshes: None)
    monkeypatch.setattr(
        monarch_system, "setup_torch_elastic_env", setup_torch_elastic_env
    )
    return spawned


def _provision(actor_system):
    asyncio.run(
        actor_system.provision(
            trainer_parallelism=ParallelismConfig(data_parallel_shard_degree=4),
            generator_parallelism=InferenceParallelismConfig(tensor_parallel_degree=2),
            num_generators=2,
        )
    )


def test_layout_round_trips_through_env(monkeypatch):
    plan = _plan(FakeLauncher.Config(), num_generators=2)
    monkeypatch.setenv(LAYOUT_ENV, plan.env[LAYOUT_ENV])

    assert Layout.from_env() == plan.layout


def _build_actor_system(config):
    """Run the controller's override with ``config`` as its ``self.config``,
    without constructing the whole controller."""
    controller = SimpleNamespace(config=config)
    return asyncio.run(MonarchControllerWithLauncher._build_actor_system(controller))


def test_launch_controller_requires_layout_env(monkeypatch):
    monkeypatch.delenv(LAYOUT_ENV, raising=False)
    config = SimpleNamespace(launcher=FakeLauncher.Config())
    with pytest.raises(ValueError, match=LAYOUT_ENV):
        _build_actor_system(config)


def test_launch_controller_builds_the_actor_system_on_acquired_hosts(
    spawned, monkeypatch
):
    plan = _plan(FakeLauncher.Config(), num_generators=2)
    monkeypatch.setenv(LAYOUT_ENV, plan.env[LAYOUT_ENV])
    events = []
    config = SimpleNamespace(
        launcher=FakeLauncher.Config(events=events),
        actor_system=MonarchActorSystem.Config(),
    )

    actor_system = _build_actor_system(config)
    _provision(actor_system)
    # The CPU meshes fail to stop; stop logs it and goes on.
    asyncio.run(actor_system.stop())

    assert events == ["acquire_hosts"]
    host_meshes = HostMeshes(
        trainer="hosts[node_0]",
        generators=["hosts[node_0]", "hosts[node_0]"],
        gpus_per_node=8,
    )
    assert actor_system._host_meshes == host_meshes
    assert spawned["host_meshes"] == host_meshes


@pytest.fixture
def slurm_job(monkeypatch):
    """Patch `monarch.job.SlurmJob`; record its kwargs, client script and env."""
    import monarch.job

    calls = {}

    class FakeSlurmJob:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def apply(self, client_script):
            calls["client_script"] = client_script
            calls["env"] = dict(os.environ)
            calls["cwd"] = os.getcwd()
            self._slurm_job_id = "42"

    monkeypatch.setattr(monarch.job, "SlurmJob", FakeSlurmJob)
    monkeypatch.delenv("SBATCH_EXPORT", raising=False)
    # submit sets these in os.environ; monkeypatch restores them afterwards.
    for name in ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "executable", "/venv/python")
    return calls


def _slurm_plan(tmp_path):
    return LaunchPlan(
        job_name="j-abcdef",
        layout=Layout(
            gpus_per_node=4,
            mesh_nodes={"trainer": 2, "generator": 1},
            role_mesh={"trainer": "trainer", "generator": "generator"},
        ),
        controller_args=["--module", "my recipes"],
        env={"FROM_PLAN": "1"},
        dump_folder=str(tmp_path),
    )


def test_slurm_submit_builds_the_job(tmp_path, slurm_job):
    launcher = SlurmLauncher.Config(
        gpus_per_node=4,
        slurm_args=["--partition=p", "--segment=1"],
        bind_to="::",
    ).build()

    assert launcher.submit(_slurm_plan(tmp_path)) == "42"
    assert slurm_job["kwargs"] == dict(
        meshes={"trainer": 2, "generator": 1},
        gpus_per_node=4,
        slurm_args=[f"--chdir={os.getcwd()}", "--partition=p", "--segment=1"],
        log_dir=str(tmp_path / "slurm"),
        job_name="j-abcdef",
        python_exe="/venv/python",
        exclusive=True,
        bind_to="::",
    )
    assert shlex.split(slurm_job["client_script"]) == [
        "/venv/python",
        "-m",
        "torchtitan.rl.train",
        "--module",
        "my recipes",
    ]
    assert slurm_job["env"]["FROM_PLAN"] == "1"


def test_slurm_submit_defaults_node_local_jit_caches(tmp_path, slurm_job, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/from/shell")
    plan = replace(_slurm_plan(tmp_path), env={"TRITON_CACHE_DIR": "/from/launcher"})

    SlurmLauncher.Config().build().submit(plan)

    env = slurm_job["env"]
    assert env["TRITON_CACHE_DIR"] == "/from/launcher"
    assert env["VLLM_CACHE_ROOT"] == "/from/shell"
    assert env["FLASHINFER_WORKSPACE_BASE"] == (
        f"/tmp/torchtitan-{os.getuid()}/flashinfer"
    )


def test_slurm_submit_writes_job_state_per_job(tmp_path, slurm_job):
    SlurmLauncher.Config().build().submit(_slurm_plan(tmp_path))

    state_dir = str(tmp_path / "launcher" / "j-abcdef")
    # Monarch writes ./.monarch/job_state.pkl relative to the apply() cwd.
    assert slurm_job["cwd"] == state_dir
    assert slurm_job["env"]["TORCHTITAN_RL_SLURM_JOB_STATE"] == (
        f"{state_dir}/.monarch/job_state.pkl"
    )


@pytest.mark.parametrize(
    "from_launcher_env", [False, True], ids=["submit_shell", "launcher_env"]
)
def test_slurm_submit_rejects_sbatch_export(
    tmp_path, slurm_job, monkeypatch, from_launcher_env
):
    plan = _slurm_plan(tmp_path)
    if from_launcher_env:
        plan = replace(plan, env={"SBATCH_EXPORT": "NONE"})
    else:
        monkeypatch.setenv("SBATCH_EXPORT", "NONE")
    with pytest.raises(ValueError, match="SBATCH_EXPORT"):
        SlurmLauncher.Config().build().submit(plan)


def test_slurm_acquire_hosts(monkeypatch):
    import monarch.job

    state = SimpleNamespace(trainer="t", generator="g")
    monkeypatch.setenv("TORCHTITAN_RL_SLURM_JOB_STATE", "/job/state.pkl")
    monkeypatch.setattr(
        monarch.job,
        "job_load",
        lambda path: SimpleNamespace(state=lambda: state)
        if path == "/job/state.pkl"
        else None,
    )
    launcher = SlurmLauncher.Config().build()
    layout = Layout(
        gpus_per_node=8, mesh_nodes={"trainer": 1, "generator": 1}, role_mesh={}
    )

    assert asyncio.run(launcher.acquire_hosts(layout)) == {
        "trainer": "t",
        "generator": "g",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gpus_per_node=0),
        dict(job_name=""),
        dict(job_name="my job"),
        dict(job_name="../x"),
        dict(slurm_args=["--nodes=2"]),
        dict(slurm_args=["-N2"]),
        dict(slurm_args=["--chdir=/x"]),
        dict(slurm_args=["--nodes 2"]),
        dict(slurm_args=["--partition=p --nodes=2"]),
        dict(slurm_args=["--export=NONE"]),
        dict(slurm_args=["--parsable"]),
        dict(slurm_args=["-Q"]),
    ],
)
def test_slurm_config_validation(kwargs):
    with pytest.raises(ValueError):
        SlurmLauncher.Config(**kwargs)
