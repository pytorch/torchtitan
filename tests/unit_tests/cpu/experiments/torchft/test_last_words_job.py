# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Exercise supervisor deadlines around a blocking checkpoint validator."""

import json
import sys
import time
from types import SimpleNamespace

import pytest
from torchtitan.last_words import job


def run_validated_save(
    tmp_path, monkeypatch, *, validation_seconds, exits, attempt_seconds=100
):
    clock = SimpleNamespace(now=0.0, published=None)
    stopped_at = []

    def sleep(seconds):
        clock.now += seconds

    class Child:
        code = None

        def poll(self):
            if (
                self.code is None
                and exits
                and clock.published is not None
                and clock.now >= clock.published + 2
            ):
                self.code = 0
            return self.code

        @property
        def returncode(self):
            return self.poll()

    child = Child()

    def stop_children(children):
        for process in children:
            if process.poll() is None:
                stopped_at.append(clock.now)
                process.code = -15

    class Election:
        finished = False

        def __init__(self, root, run_id, **kwargs):
            self.control = root / "last_words/control" / run_id
            self.control.mkdir(parents=True)

        def tick(self, children, *, terminate):
            if not self.finished:
                sleep(validation_seconds)
                clock.published = clock.now
                (self.control / "result.json").write_text(
                    json.dumps(dict(success=True))
                )
                self.finished = True

    monkeypatch.setattr(
        job,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=sleep,
            strftime=time.strftime,
            gmtime=time.gmtime,
        ),
    )
    monkeypatch.setattr(job, "PROTOCOL", {"SaveElection": Election})
    monkeypatch.setattr(job, "start_child", lambda *args, **kwargs: child)
    monkeypatch.setattr(job, "stop_children", stop_children)
    state_dir = tmp_path / "state"
    code = job.supervised_attempts(
        dict(
            coordination_backend="file",
            commands=[[sys.executable, "-c", "pass", "{run_id}", "{resume_path}"]],
            checkpoint_root=str(tmp_path / "checkpoint"),
            required_free_bytes=4096,
            max_attempts=1,
            max_attempt_seconds=attempt_seconds,
            no_progress_seconds=100,
        ),
        state_dir,
        lambda deadline, children: None,
    )
    result = json.loads(next(state_dir.glob("launch-*.json")).read_text())
    return code, result, stopped_at, clock


@pytest.mark.parametrize("validation_seconds", [0, 6])
def test_slow_validation_does_not_consume_terminal_exit_window(
    tmp_path, monkeypatch, validation_seconds
):
    code, result, stopped_at, _ = run_validated_save(
        tmp_path,
        monkeypatch,
        validation_seconds=validation_seconds,
        exits=True,
    )
    assert code == 0 and result["success"]
    assert result["return_codes"] == [0]
    assert stopped_at == []


@pytest.mark.parametrize("validation_seconds", [0, 6])
def test_hung_launcher_still_stops_five_seconds_after_validation(
    tmp_path, monkeypatch, validation_seconds
):
    code, result, stopped_at, clock = run_validated_save(
        tmp_path,
        monkeypatch,
        validation_seconds=validation_seconds,
        exits=False,
    )
    assert code == 1 and not result["success"]
    assert result["return_codes"] == [-15]
    assert stopped_at == [clock.published + 5]


def test_validation_does_not_extend_whole_attempt_deadline(tmp_path, monkeypatch):
    code, result, _, clock = run_validated_save(
        tmp_path,
        monkeypatch,
        validation_seconds=6,
        exits=True,
        attempt_seconds=4,
    )
    assert code == 1 and not result["success"]
    assert result["forced_reason"] == "external_deadline"
    assert clock.now == 6
