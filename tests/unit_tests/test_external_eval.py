# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from torchtitan.components.external_eval import ExternalEval


def test_external_eval_close_waits_for_results():
    external_eval = ExternalEval(ExternalEval.Config())
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    process.pid = 123
    external_eval.processes = [process]

    external_eval.close()

    process.wait.assert_called_once_with()
    assert external_eval.processes == []


def test_external_eval_request_only_queues_command(tmp_path):
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True,
            path="/package/run_external_eval.py",
            tasks="task",
            request_only=True,
            eval_cuda_visible_devices="0,1",
        )
    )
    trainer_config = SimpleNamespace(
        dump_folder=str(tmp_path),
        hf_assets_path="/package/assets",
        model_spec=SimpleNamespace(name="olmo3", flavor="7B"),
    )

    external_eval.launch(
        step=4000,
        checkpoint_dir="/checkpoints/ema/avg-4/step-4000",
        output_name="step-4000-averaged-4",
        source_steps=[1000, 2000, 3000, 4000],
        trainer_config=trainer_config,
    )

    output_dir = tmp_path / "eval" / "step-4000-averaged-4"
    with (output_dir / "eval_request.json").open() as f:
        request = json.load(f)

    assert request["command"][0] == sys.executable
    assert request["command"][1] == "/package/run_external_eval.py"
    assert request["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert not (output_dir / "launch.log").exists()

    external_eval.close()
    assert (tmp_path / "eval" / "_TRAINING_COMPLETE").exists()
