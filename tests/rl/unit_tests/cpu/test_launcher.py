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
    MonarchLauncher,
)
from torchtitan.rl.actors.monarch.launcher.slurm import SlurmLauncher
from torchtitan.rl.actors.monarch.provisioning import HostMeshes
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig


class FakeLauncher(MonarchLauncher):
    @dataclass(kw_only=True, slots=True)
    class Config(MonarchLauncher.Config):
        in_job: bool = True
        events: list = field(default_factory=list)

    def gpus_per_node(self) -> int:
        return 8

    def submit(self, plan: LaunchPlan) -> str:
        self.config.events.append("submit")
        return "job-123"

    def in_job(self) -> bool:
        return self.config.in_job

    async def acquire_hosts(self, layout: Layout) -> dict:
        self.config.events.append("acquire_hosts")
        return {name: f"hosts[{name}]" for name in layout.mesh_nodes}

    async def release_hosts(self, meshes: dict) -> None:
        self.config.events.append("release_hosts")
        raise RuntimeError("release failed")


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


@pytest.mark.parametrize(
    "sizes, mesh_nodes",
    [
        ((4, 2, 2), {"trainer": 1, "generator_0": 1, "generator_1": 1}),
        ((16, 8, 1), {"trainer": 2, "generator": 1}),
        ((16, 16, 2), {"trainer": 2, "generator_0": 2, "generator_1": 2}),
    ],
)
def test_layout(sizes, mesh_nodes):
    trainer_dp, generator_tp, num_generators = sizes
    layout = _plan(
        FakeLauncher.Config(),
        trainer_dp=trainer_dp,
        generator_tp=generator_tp,
        num_generators=num_generators,
    ).layout

    assert layout.mesh_nodes == mesh_nodes
    assert layout.role_mesh == {role: role for role in mesh_nodes}


def test_layout_rejects_partial_multi_node_role():
    with pytest.raises(ValueError, match=r"trainer world size \(12\)"):
        _plan(FakeLauncher.Config(), trainer_dp=12, generator_tp=8)


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


def test_split_args():
    assert submit._split_args(["--module", "m", "--dry-run", "--override", "x"]) == (
        True,
        ["--module", "m", "--override", "x"],
    )


def _run_submit(monkeypatch, config, argv):
    monkeypatch.setattr(submit.ConfigLoader, "load", lambda self, args: config)
    # The fake config stands in for a Controller.Config.
    monkeypatch.setattr(submit, "Controller", SimpleNamespace(Config=SimpleNamespace))
    return submit.main(argv)


def _with_launcher(tmp_path, launcher):
    return _rl_config(
        dump_folder=str(tmp_path),
        actor_system=MonarchActorSystem.Config(launcher=launcher),
    )


def test_submit_dry_run_does_not_submit(monkeypatch, tmp_path, capsys):
    events = []
    config = _with_launcher(tmp_path, FakeLauncher.Config(events=events))

    assert _run_submit(monkeypatch, config, ["--dry-run", "--module", "m"]) is None
    assert events == []
    assert "mesh trainer: 1 node(s), roles trainer" in capsys.readouterr().out


def test_submit_plans_after_overrides(monkeypatch, tmp_path, capsys):
    config = _with_launcher(tmp_path, FakeLauncher.Config())

    def resolve_overrides():
        # Stands in for an override that adds a generator.
        config.num_generators = 2

    config.resolve_overrides = resolve_overrides

    _run_submit(monkeypatch, config, ["--dry-run", "--module", "m"])

    assert "mesh generator_1: 1 node(s), roles generator_1" in capsys.readouterr().out


def test_submit_submits_and_returns_the_job_handle(monkeypatch, tmp_path):
    events = []
    config = _with_launcher(tmp_path, FakeLauncher.Config(events=events))

    assert _run_submit(monkeypatch, config, ["--module", "m"]) == "job-123"
    assert events == ["submit"]


def test_submit_requires_a_launcher(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="does not set"):
        _run_submit(monkeypatch, _with_launcher(tmp_path, None), ["--module", "m"])


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
        monarch_system, "spawn_cpu_procs", lambda n: SimpleNamespace(stop=_stop_fails)
    )
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


def test_provision_without_launcher_runs_on_this_host(spawned):
    _provision(MonarchActorSystem.Config().build())
    assert spawned["host_meshes"] is None


def test_provision_raises_outside_the_job(spawned):
    launcher = FakeLauncher.Config(in_job=False)
    with pytest.raises(ValueError, match="launcher.submit"):
        _provision(MonarchActorSystem.Config(launcher=launcher).build())


def test_layout_round_trips_through_env(monkeypatch):
    plan = _plan(FakeLauncher.Config(), num_generators=2)
    monkeypatch.setenv(LAYOUT_ENV, plan.env[LAYOUT_ENV])

    assert Layout.from_env() == plan.layout


def test_provision_requires_layout_env(spawned, monkeypatch):
    monkeypatch.delenv(LAYOUT_ENV, raising=False)
    with pytest.raises(ValueError, match=LAYOUT_ENV):
        _provision(MonarchActorSystem.Config(launcher=FakeLauncher.Config()).build())


def test_provision_acquires_and_stop_releases_hosts(spawned, monkeypatch):
    plan = _plan(FakeLauncher.Config(), num_generators=2)
    monkeypatch.setenv(LAYOUT_ENV, plan.env[LAYOUT_ENV])
    events = []
    launcher = FakeLauncher.Config(events=events)
    actor_system = MonarchActorSystem.Config(launcher=launcher).build()

    _provision(actor_system)
    # The CPU meshes fail to stop and release_hosts raises; stop logs both.
    asyncio.run(actor_system.stop())
    asyncio.run(actor_system.stop())

    assert spawned["host_meshes"] == HostMeshes(
        trainer="hosts[trainer]",
        generators=["hosts[generator_0]", "hosts[generator_1]"],
        gpus_per_node=8,
    )
    assert events == ["acquire_hosts", "release_hosts"]


def test_passed_host_meshes_take_precedence(spawned):
    launcher = FakeLauncher.Config(in_job=False)
    _provision(MonarchActorSystem.Config(launcher=launcher).build(host_meshes="mine"))
    assert spawned["host_meshes"] == "mine"


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


def test_slurm_submit_can_share_nodes(tmp_path, slurm_job):
    SlurmLauncher.Config(exclusive=False).build().submit(_slurm_plan(tmp_path))
    assert slurm_job["kwargs"]["exclusive"] is False


def test_slurm_submit_writes_job_state_per_job(tmp_path, slurm_job):
    SlurmLauncher.Config().build().submit(_slurm_plan(tmp_path))

    state_dir = str(tmp_path / "launcher" / "j-abcdef")
    # Monarch writes ./.monarch/job_state.pkl relative to the apply() cwd.
    assert slurm_job["cwd"] == state_dir
    assert slurm_job["env"]["TORCHTITAN_RL_SLURM_JOB_STATE"] == (
        f"{state_dir}/.monarch/job_state.pkl"
    )


def test_slurm_submit_rejects_sbatch_export(tmp_path, slurm_job, monkeypatch):
    monkeypatch.setenv("SBATCH_EXPORT", "NONE")
    with pytest.raises(ValueError, match="SBATCH_EXPORT"):
        SlurmLauncher.Config().build().submit(_slurm_plan(tmp_path))


def test_slurm_submit_rejects_sbatch_export_from_launcher_env(tmp_path, slurm_job):
    plan = replace(_slurm_plan(tmp_path), env={"SBATCH_EXPORT": "NONE"})
    with pytest.raises(ValueError, match="SBATCH_EXPORT"):
        SlurmLauncher.Config().build().submit(plan)


def test_slurm_acquire_hosts_and_in_job(monkeypatch):
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
    monkeypatch.setenv("MONARCH_BATCH_JOB", "1")
    assert launcher.in_job()
    monkeypatch.delenv("MONARCH_BATCH_JOB")
    assert not launcher.in_job()


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


def test_slurm_recipe_plans_a_node_per_role():
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_flex_slurm

    config = rl_grpo_qwen3_0_6b_flex_slurm()
    plan = config.actor_system.launcher.build().plan(config, [])

    assert plan.layout.mesh_nodes == {"trainer": 1, "generator_0": 1, "generator_1": 1}
