# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import SimpleNamespace

from torchtitan.experiments.evals.wandb_logger import EvalWandBLogger, flatten_results

RECORD = {
    "results": {
        "hellaswag": {
            "alias": "hellaswag",
            "acc,none": 0.25,
            "acc_stderr,none": 0.01,
            "acc_norm,none": 0.3,
        },
        "gsm8k": {"exact_match,strict-match": 0.1},
        "trainstation_mmlu": {"acc,none": 0.4, "acc_stderr,none": "N/A"},
    }
}


class _FakeRun:
    def __init__(self):
        self.defined = []
        self.logged = []

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, data):
        self.logged.append(data)

    def finish(self):
        pass


def _install_fake_wandb(monkeypatch):
    calls = {}
    run = _FakeRun()

    def init(**kwargs):
        calls.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return calls, run


def test_flatten_results():
    assert flatten_results(RECORD) == {
        "eval/hellaswag/acc": 0.25,
        "eval/hellaswag/acc_stderr": 0.01,
        "eval/hellaswag/acc_norm": 0.3,
        "eval/gsm8k/exact_match_strict-match": 0.1,
        "eval/trainstation_mmlu/acc": 0.4,
    }


def test_logger_plots_against_train_step(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_RUN_NAME", "run7")
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    calls, run = _install_fake_wandb(monkeypatch)

    output_folder = str(tmp_path / "run7" / "evals")
    logger = EvalWandBLogger(output_folder=output_folder, suite="quick", config={})
    logger.log(300, RECORD)

    assert calls["name"] == "run7-eval-quick"
    assert calls["group"] == "run7"
    assert calls["resume"] == "allow"
    assert ("eval/*", "train_step") in run.defined
    assert run.logged == [{"train_step": 300, **flatten_results(RECORD)}]

    # A restarted job resumes the same run; another suite gets its own.
    restarted, _ = _install_fake_wandb(monkeypatch)
    EvalWandBLogger(output_folder=output_folder, suite="quick", config={})
    assert restarted["id"] == calls["id"]
    other_suite, _ = _install_fake_wandb(monkeypatch)
    EvalWandBLogger(output_folder=output_folder, suite="full", config={})
    assert other_suite["id"] != calls["id"]
