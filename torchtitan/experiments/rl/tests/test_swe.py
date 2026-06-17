# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the SWE (R2E-Gym) example: grading, agent-loop env, rubric.

No GPU and no container: a fake in-process sandbox stands in for podman.
"""

import asyncio
import json
import os

from renderers.base import ParsedToolCall, ToolCallParseStatus

from torchtitan.experiments.rl.examples.swe import (
    grading,
    R2EGymDataset,
    R2EGymSample,
    RewardR2EGym,
    SweEnv,
)
from torchtitan.experiments.rl.examples.swe.sandbox import ExecResult
from torchtitan.experiments.rl.rollout import Rollout, RolloutStatus, RolloutTurn


# --------------------------------------------------------------------------- #
# grading (pure)
# --------------------------------------------------------------------------- #
_JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite>
  <testcase classname="pkg.TestA" name="test_pass"></testcase>
  <testcase classname="pkg.TestA" name="test_fail"><failure>boom</failure></testcase>
  <testcase classname="pkg.TestA" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def test_parse_junit_statuses():
    parsed = grading.parse_junit(_JUNIT)
    assert parsed["test_pass"] == "PASSED"
    assert parsed["test_fail"] == "FAILED"
    assert parsed["test_skip"] == "SKIPPED"
    # indexed under Class.name too
    assert parsed["TestA.test_pass"] == "PASSED"


def test_parse_junit_bad_xml_is_empty():
    assert grading.parse_junit("not xml") == {}


def test_is_resolved_requires_all_expected():
    parsed = {"test_pass": "PASSED", "test_fail": "FAILED"}
    assert grading.is_resolved(parsed, {"test_pass": "PASSED"})
    assert grading.is_resolved(parsed, {"test_pass": "PASSED", "test_fail": "FAILED"})
    # one expected status not reproduced -> not resolved
    assert not grading.is_resolved(parsed, {"test_fail": "PASSED"})
    # empty expected is never "resolved"
    assert not grading.is_resolved(parsed, {})


def test_matched_fraction():
    parsed = {"a": "PASSED", "b": "FAILED"}
    assert grading.matched_fraction(parsed, {"a": "PASSED", "b": "PASSED"}) == 0.5
    assert grading.matched_fraction(parsed, {"a": "PASSED", "b": "FAILED"}) == 1.0


# --------------------------------------------------------------------------- #
# fake sandbox + env
# --------------------------------------------------------------------------- #
class FakeSandbox:
    """In-process stand-in for a container sandbox.

    Records exec/write calls; serves a canned junit report so grading runs end to
    end without podman. ``exec`` returns ``junit_exit`` for the pytest command.
    """

    def __init__(self, junit_xml: str, junit_exit: int = 0):
        self._junit_xml = junit_xml
        self._junit_exit = junit_exit
        self.exec_cmds: list[str] = []
        self.written: dict[str, str] = {}
        self.closed = False

    async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
        self.exec_cmds.append(cmd)
        if "pytest" in cmd:
            return ExecResult(output="1 passed", exit_code=self._junit_exit)
        if cmd.startswith("find "):  # a no-match find succeeds with empty output
            return ExecResult(output="", exit_code=0)
        if cmd.startswith("cat "):
            path = cmd[len("cat ") :].strip().strip("'")
            return ExecResult(output=self.written.get(path, ""), exit_code=0)
        return ExecResult(output=f"ran: {cmd}", exit_code=0)

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content

    async def read_file(self, path: str) -> str:
        if "junit" in path:
            return self._junit_xml
        return self.written.get(path, "")

    async def close(self) -> None:
        self.closed = True


class FakeFactory:
    """Duck-typed SandboxFactory: hands every env the same fake sandbox."""

    def __init__(self, sandbox: FakeSandbox, repo_root: str = "/testbed"):
        self._sandbox = sandbox
        self.repo_root = repo_root

    async def provision(self, *, image: str) -> FakeSandbox:
        del image
        return self._sandbox


_SAMPLE = R2EGymSample(
    instance_id="demo-1",
    image="demo:latest",
    problem_statement="Fix the widget so it renders.",
    test_file_names=("test_widget.py",),
    test_file_codes=("def test_pass():\n    assert True\n",),
    expected_output_json=json.dumps({"test_pass": "PASSED"}),
)


def _bash(cmd: str) -> ParsedToolCall:
    return ParsedToolCall(raw="", name="bash", arguments={"cmd": cmd})


def _submit() -> ParsedToolCall:
    return ParsedToolCall(raw="", name="submit", arguments={})


def _make_env(sandbox: FakeSandbox, **cfg) -> SweEnv:
    env = SweEnv(
        SweEnv.Config(**cfg), env_input=_SAMPLE, sandbox_factory=FakeFactory(sandbox)
    )
    asyncio.run(env.init())  # provisions the fake sandbox
    return env


def test_env_init_exposes_bash_and_submit():
    sb = FakeSandbox(_JUNIT)
    env = SweEnv(SweEnv.Config(), env_input=_SAMPLE, sandbox_factory=FakeFactory(sb))
    out = asyncio.run(env.init())
    assert [t["name"] for t in out.tools] == ["bash", "submit"]
    assert "Fix the widget" in out.init_prompt_messages[0]["content"]


def test_env_bash_call_returns_observation():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [_bash("ls")]}))
    assert out.done is False
    assert len(out.env_messages) == 1
    assert out.env_messages[0]["role"] == "tool"
    assert "exit=0" in out.env_messages[0]["content"]
    assert "ran: ls" in out.env_messages[0]["content"]


def test_env_empty_output_is_labeled():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    out = asyncio.run(
        env.step({"role": "assistant", "tool_calls": [_bash("find / -name nope")]})
    )
    assert out.env_messages[0]["content"] == "(exit=0)\n(no output)"


def test_env_submit_grades_and_terminates():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [_submit()]}))
    assert out.done is True
    assert out.env_rewards["resolved"] == 1.0
    assert out.env_rewards["eval_ran"] == 1.0
    # the canonical hidden test was injected at grade time
    assert "/testbed/test_widget.py" in sb.written


def test_env_no_tool_call_grades_and_terminates():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    out = asyncio.run(env.step({"role": "assistant", "content": "done"}))
    assert out.done is True
    assert out.env_rewards["resolved"] == 1.0


def test_env_max_turns_forces_grading():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb, max_turns=1)
    # first (and only allowed) turn: even a bash call terminates + grades
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [_bash("ls")]}))
    assert out.done is True
    assert out.env_rewards["resolved"] == 1.0


def test_env_unresolved_when_tests_fail():
    # expected says test_pass should PASS, but junit reports it FAILED
    failing = _JUNIT.replace(
        '<testcase classname="pkg.TestA" name="test_pass"></testcase>',
        '<testcase classname="pkg.TestA" name="test_pass"><failure>x</failure></testcase>',
    )
    sb = FakeSandbox(failing)
    env = _make_env(sb)
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [_submit()]}))
    assert out.env_rewards["resolved"] == 0.0
    assert out.env_rewards["passed_frac"] == 0.0


def test_env_malformed_tool_call_is_reported_not_fatal():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    bad = ParsedToolCall(
        raw="<garbage>",
        name=None,
        arguments=None,
        status=ToolCallParseStatus.INVALID_JSON,
    )
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [bad]}))
    assert out.done is False
    assert "malformed" in out.env_messages[0]["content"]


def test_env_close_is_idempotent():
    sb = FakeSandbox(_JUNIT)
    env = _make_env(sb)
    asyncio.run(env.close())
    assert sb.closed
    asyncio.run(env.close())  # second close must not raise


# --------------------------------------------------------------------------- #
# rubric
# --------------------------------------------------------------------------- #
def _rollout(env_rewards: dict | None, status=RolloutStatus.COMPLETED) -> Rollout:
    turns = [
        RolloutTurn(
            prompt_token_ids=[1],
            completion_token_ids=[2],
            completion_logprobs=[-0.1],
            completion_message={"role": "assistant", "content": "x"},
            env_rewards=env_rewards or {},
        )
    ]
    return Rollout(group_id="g", sample_id="s", status=status, turns=turns)


def test_reward_reads_resolved():
    r = RewardR2EGym(RewardR2EGym.Config())
    got = asyncio.run(r(_rollout({"resolved": 1.0, "passed_frac": 1.0}), _SAMPLE))
    assert got == 1.0


def test_reward_zero_when_unresolved_by_default():
    r = RewardR2EGym(RewardR2EGym.Config())
    got = asyncio.run(r(_rollout({"resolved": 0.0, "passed_frac": 0.5}), _SAMPLE))
    assert got == 0.0


def test_reward_partial_credit_opt_in():
    r = RewardR2EGym(RewardR2EGym.Config(use_partial_credit=True))
    got = asyncio.run(r(_rollout({"resolved": 0.0, "passed_frac": 0.5}), _SAMPLE))
    assert got == 0.5


def test_reward_zero_when_never_graded():
    # a rollout truncated before grading has no 'resolved' in any turn
    r = RewardR2EGym(RewardR2EGym.Config())
    got = asyncio.run(r(_rollout(None, status=RolloutStatus.TRUNCATED_LENGTH), _SAMPLE))
    assert got == 0.0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def test_dataset_loads_bundled_smoke_jsonl():
    here = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(here, "examples", "swe", "data", "r2e_smoke.jsonl")
    ds = R2EGymDataset(R2EGymDataset.Config(data_path=path, shuffle=False))
    sample = next(iter(ds))
    assert isinstance(sample, R2EGymSample)
    assert sample.image.startswith("docker.io/namanjain12/orange3_final")
    assert len(sample.test_file_names) == len(sample.test_file_codes)
