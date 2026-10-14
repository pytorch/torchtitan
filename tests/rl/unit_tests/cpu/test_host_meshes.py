# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the multi-host plan in torchtitan.rl.train."""

from types import SimpleNamespace

import pytest

from torchtitan.rl import train


class FakeMesh:
    def __init__(self, hosts):
        self.hosts = hosts

    def __len__(self):
        return len(self.hosts)

    def slice(self, *, hosts):
        return FakeMesh(self.hosts[hosts])


def test_one_host_is_shared_by_every_role():
    hosts = FakeMesh(["a"])
    plan = train.plan_host_meshes(
        hosts,
        trainer_world_size=2,
        per_generator_world_size=1,
        num_generators=2,
        gpus_per_node=4,
    )
    assert plan.trainer is hosts and all(m is hosts for m in plan.generators)


def test_trainer_takes_the_first_hosts_and_generators_share_their_node():
    plan = train.plan_host_meshes(
        FakeMesh(["a", "b", "c", "d"]),
        trainer_world_size=8,
        per_generator_world_size=1,
        num_generators=6,
        gpus_per_node=4,
    )
    assert plan.trainer.hosts == ["a", "b"]
    assert [m.hosts for m in plan.generators] == [["c"]] * 4 + [["d"]] * 2
    assert (
        len({id(m) for m in plan.generators[:4]}) == 1
    )  # same object -> disjoint GPUs
    assert plan.generators[0] is not plan.generators[4]


def test_plan_rejects_too_few_hosts_and_oversized_generators():
    with pytest.raises(ValueError, match="the job has 2"):
        train.plan_host_meshes(
            FakeMesh(["a", "b"]),
            trainer_world_size=8,
            per_generator_world_size=1,
            num_generators=4,
            gpus_per_node=4,
        )
    with pytest.raises(ValueError, match="a node has 4"):
        train.plan_host_meshes(
            FakeMesh(["a", "b"]),
            trainer_world_size=4,
            per_generator_world_size=8,
            num_generators=1,
            gpus_per_node=4,
        )


def test_no_job_env_means_single_host(monkeypatch):
    monkeypatch.delenv(train.MONARCH_JOB_ENV, raising=False)
    # Without a job the config is never read.
    assert train._job_host_meshes(SimpleNamespace()) is None
