# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the candidate ``env`` lever.

The agent may set process environment variables on the training subprocess; the
harness applies them BEFORE launch (before ``init_process_group``) so runtime
knobs like NCCL_*/CUDA_* are reachable. The admissibility guard rejects env that
would game the harness itself (the verify run's determinism/seed) or redefine the
run (MODULE/CONFIG/NGPU/PYTHONPATH); everything else flows through and is
adjudicated by the faithfulness gate like any other change.
"""

from __future__ import annotations

import json
import os
import tempfile

from torchtitan_autoresearch import gate as G, workload_guard as wg
from torchtitan_autoresearch.agent import SingleTurnLLMAgent
from torchtitan_autoresearch.constitution import load_constitution
from torchtitan_autoresearch.executor import FakeExecutor, SubprocessExecutor
from torchtitan_autoresearch.ledger import Ledger, Record
from torchtitan_autoresearch.state import HarnessState
from torchtitan_autoresearch.types import Candidate, Observation

_CONST = os.path.join(os.path.dirname(__file__), "Constitution.md")


def _rules():
    return load_constitution(_CONST)


def test_run_key_is_env_sensitive():
    ex = SubprocessExecutor(
        ".", log_freq=1, ngpu=1, module="llama3", config="llama3_8b"
    )
    base = Candidate(label="x", command=["--a=1"])
    e16 = Candidate(label="x", command=["--a=1"], env={"NCCL_MIN_NCHANNELS": "16"})
    e32 = Candidate(label="x", command=["--a=1"], env={"NCCL_MIN_NCHANNELS": "32"})
    assert ex._key(base) != ex._key(e16) != ex._key(e32)


def test_admissible_allows_runtime_env():
    ok, _ = wg.admissible(
        Candidate(
            label="n",
            env={"NCCL_MIN_NCHANNELS": "16", "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        ),
        _rules(),
    )
    assert ok


def test_admissible_rejects_gaming_env():
    rules = _rules()
    for env in (
        {"PYTHONHASHSEED": "0"},  # seed
        {"MY_DETERMINISTIC_FLAG": "1"},  # name-pattern determinism
        {"CUBLAS_WORKSPACE_CONFIG": ":0:0"},  # determinism governance
        {"CONFIG": "llama3_70b"},  # redefines the run
        {"NGPU": "1"},  # redefines the regime
    ):
        ok, reason = wg.admissible(Candidate(label="bad", env=env), rules)
        assert not ok and reason


def test_env_merge_harness_keys_win():
    # Candidate env is applied, but harness operational keys override it.
    c = Candidate(label="t", env={"NCCL_BUFFSIZE": "8388608", "CONFIG": "evil"})
    env = {
        **os.environ,
        **(c.env or {}),
        "NGPU": "4",
        "MODULE": "llama3",
        "CONFIG": "llama3_8b",
    }
    assert env["NCCL_BUFFSIZE"] == "8388608"
    assert env["CONFIG"] == "llama3_8b"


def test_ledger_round_trips_env():
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(os.path.join(d, "l.tsv"))
        led.append(
            Record(
                commit="abc1234",
                label="nccl",
                env={"NCCL_MIN_NCHANNELS": "16", "NCCL_BUFFSIZE": "8388608"},
            )
        )
        assert led.read()[0]["env"] == "NCCL_BUFFSIZE=8388608;NCCL_MIN_NCHANNELS=16"


def test_agent_parses_and_dedups_env():
    prompts: list[str] = []

    def ask(p):
        prompts.append(p)
        n = len(prompts)
        if n <= 2:  # first two replies are the SAME (command+env) -> dedup, retry
            return json.dumps(
                {
                    "label": "a",
                    "command": ["--x=1"],
                    "env": {"NCCL_MIN_NCHANNELS": "16"},
                }
            )
        return json.dumps(
            {"label": "b", "command": ["--x=1"], "env": {"NCCL_MIN_NCHANNELS": "32"}}
        )

    a = SingleTurnLLMAgent(ask=ask, repo_root="/nonexistent")
    a.start()
    rules = {
        "objective": {},
        "quality": {},
        "workload": {},
        "editable": {"files": []},
        "banned_workload_fields": [],
        "fixed_fields": [],
    }
    obs = Observation(
        rules=rules,
        ledger=[],
        champion=None,
        golden=None,
        deferred_families=[],
        ideas=[],
    )
    c1 = a.propose(obs)
    c2 = a.propose(obs)
    assert c1.env == {"NCCL_MIN_NCHANNELS": "16"}
    assert c2.env == {
        "NCCL_MIN_NCHANNELS": "32"
    }  # same cmd+env deduped -> got distinct
    assert "NCCL_MIN_NCHANNELS" in prompts[0]  # the lever is advertised


def _fresh_state(tmp: str, champ_tps: float) -> str:
    sf = os.path.join(tmp, "s.json")
    HarnessState(
        golden_commit="g",
        golden_det_losses=[1.0] * 8,
        golden_det_grad_norms=[1.0] * 8,
        loss_band=1e-2,
        grad_band=1e-2,
        champion_commit="g",
        champion_tps=[champ_tps],
        tps_cv=0.02,
        tps_tail_pct=3.7,
    ).save(sf)
    return sf


class _RecordingFake(FakeExecutor):
    """FakeExecutor that records the env handed to profile()."""

    def __init__(self, specs):
        super().__init__(specs)
        self.profile_env = None

    def profile(self, command=None, env=None):
        self.profile_env = env
        return ""


def test_gate_rejects_protected_env_before_running():
    with tempfile.TemporaryDirectory() as tmp:
        sf = _fresh_state(tmp, 800.0)
        led = Ledger(os.path.join(tmp, "a.tsv"))
        v = G.gate(
            Candidate(label="bad", env={"PYTHONHASHSEED": "0"}),
            rules=_rules(),
            state=HarnessState.load(sf),
            ledger=led,
            executor=_RecordingFake({}),
            statefile=sf,
        )
        assert v.status == "invalid" and "PYTHONHASHSEED" in v.detail


def test_gate_threads_env_and_promotes_faithful():
    with tempfile.TemporaryDirectory() as tmp:
        sf = _fresh_state(tmp, 800.0)
        led = Ledger(os.path.join(tmp, "b.tsv"))
        ex = _RecordingFake({"nccl": {"tps": 1100.0, "cv": 0.02, "faithful": True}})
        v = G.gate(
            Candidate(label="nccl", family="nccl", env={"NCCL_MIN_NCHANNELS": "16"}),
            rules=_rules(),
            state=HarnessState.load(sf),
            ledger=led,
            executor=ex,
            statefile=sf,
        )
        assert v.status == "keep" and v.verify == "faithful"
        assert ex.profile_env == {"NCCL_MIN_NCHANNELS": "16"}
        assert led.read()[-1]["env"] == "NCCL_MIN_NCHANNELS=16"


def test_gate_rejects_fast_but_affecting_env():
    with tempfile.TemporaryDirectory() as tmp:
        sf = _fresh_state(tmp, 800.0)
        led = Ledger(os.path.join(tmp, "c.tsv"))
        ex = _RecordingFake({"nccl2": {"tps": 1100.0, "cv": 0.02, "faithful": False}})
        v = G.gate(
            Candidate(label="nccl2", family="nccl", env={"NCCL_ALGO": "Tree"}),
            rules=_rules(),
            state=HarnessState.load(sf),
            ledger=led,
            executor=ex,
            statefile=sf,
        )
        assert v.status == "discard" and v.verify == "affecting"
