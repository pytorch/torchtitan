# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch
import argparse

from datasets import Dataset, DatasetDict
from lm_eval.tasks import TaskManager

from scripts.olmo3_harness_metrics import (
    process_gold_bpb_results,
    process_mc_results,
)
from scripts.olmo3_harness_tasks import (
    OLMO3_TASK_ALIASES,
    SKIPPED_OLMO3_TASKS,
    SUPPORTED_OLMO3_TASKS,
    prepare_task_configs,
)
from scripts.run_external_eval import build_lm_eval_command, select_task_names


def test_mc_results_include_raw_accuracy_and_bpb():
    doc = {
        "_olmo3_gold": 1,
        "_olmo3_choices": ["A", "B", "C"],
    }

    metrics = process_mc_results(
        doc,
        [(-2.0, False), (-0.5, True), (-1.0, False)],
    )

    assert metrics["acc_raw"] == 1.0
    assert metrics["bpb_v1"] == (-0.5, 1)
    assert metrics["bpb_v2"] == (-0.5, 2)


def test_gold_results_include_both_bpb_normalizations():
    metrics = process_gold_bpb_results(
        {"_olmo3_target": "    answer"},
        [(-7.0, False)],
    )

    assert metrics["bpb_v1"] == (-7.0, 9)
    assert metrics["bpb_v2"] == (-7.0, 10)


def test_generated_tasks_are_registered_and_missing_tasks_are_skipped(tmp_path):
    task_path = prepare_task_configs(tmp_path / "tasks")
    task_manager = TaskManager(include_path=str(task_path))

    for task_name in SUPPORTED_OLMO3_TASKS:
        assert task_name in task_manager.all_tasks

    assert len(SKIPPED_OLMO3_TASKS) == 10
    assert "copycolors_10way_fast" in SKIPPED_OLMO3_TASKS
    assert len(OLMO3_TASK_ALIASES) == 16
    assert set(OLMO3_TASK_ALIASES.values()) <= set(SUPPORTED_OLMO3_TASKS)
    assert "mt_mbpp_cpp_gold_bpb_3shot" in SKIPPED_OLMO3_TASKS


def test_generated_arc_task_loads_custom_metrics(tmp_path):
    rows = [
        {
            "id": str(index),
            "question": f"Question {index}",
            "choices": {
                "text": ["one", "two", "three", "four"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        }
        for index in range(6)
    ]
    dataset = Dataset.from_list(rows)
    dataset_dict = DatasetDict(train=dataset, test=dataset)
    task_path = prepare_task_configs(tmp_path / "tasks")

    with patch("datasets.load_dataset", return_value=dataset_dict):
        loaded = TaskManager(include_path=str(task_path)).load(
            ["olmo3_arc_challenge_5shot"]
        )
        task = loaded["tasks"]["olmo3_arc_challenge_5shot"]

    assert set(task._aggregation_list) == {"acc_raw", "bpb_v1", "bpb_v2"}
    assert task.doc_to_text(task.eval_docs[0]) == (
        "Question: Question 0\n" " A. one\n B. two\n C. three\n D. four\n" "Answer:"
    )


def test_two_gpu_eval_uses_torchrun_and_replicated_data_parallel(tmp_path):
    args = argparse.Namespace(
        checkpoint_dir="checkpoint",
        hf_assets_path="assets",
        model_name="olmo3",
        model_flavor="olmo3_7b",
        tasks="olmo3_arc_easy_5shot",
        output_dir=str(tmp_path),
        export_dtype="bfloat16",
        eval_gpus=2,
        batch_size=1,
        max_sequence_length=8192,
        attn_backend="varlen",
        lm_eval_bin="python -m lm_eval",
        lm_eval_model="pytorch_dcp",
        lm_eval_extra_args="",
    )

    command = build_lm_eval_command(args, task_config_path=tmp_path / "tasks")
    model_args = command[command.index("--model_args") + 1]

    assert command[:9] == [
        "torchrun",
        "--nproc_per_node",
        "2",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "-m",
        "lm_eval",
    ]
    assert "devices=2" in model_args
    assert "data_parallel_replicate_degree=2" in model_args
    assert "data_parallel_shard_degree=1" in model_args
    assert "batch_size=" not in model_args
    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--output_path") + 1] == str(tmp_path / "results.json")


def test_unsupported_olmo_tasks_are_skipped():
    selected, skipped = select_task_names("olmo3_arc_easy_5shot,copycolors_10way_fast")

    assert selected == ["olmo3_arc_easy_5shot"]
    assert skipped == {"copycolors_10way_fast": "No stock LM Harness task or dataset."}
