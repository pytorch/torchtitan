# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from scripts.run_external_eval_worker import pending_requests


def test_pending_requests_are_ordered_by_step_and_skip_results(tmp_path):
    step_10000 = tmp_path / "step-10000-merged-4"
    step_9000 = tmp_path / "step-9000-merged-4"
    step_8000 = tmp_path / "step-8000-merged-4"
    for output_dir in (step_10000, step_9000, step_8000):
        output_dir.mkdir()
        (output_dir / "eval_request.json").touch()
    (step_8000 / "eval_result.json").touch()

    assert pending_requests(tmp_path) == [
        step_9000 / "eval_request.json",
        step_10000 / "eval_request.json",
    ]
