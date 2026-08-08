# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
from pathlib import Path


SUPPORTED_OLMO3_TASKS = (
    "olmo3_arc_challenge_5shot",
    "olmo3_arc_easy_5shot",
    "olmo3_hellaswag_5shot",
    "olmo3_mmlu_humanities_5shot",
    "olmo3_mmlu_other_5shot",
    "olmo3_mmlu_social_sciences_5shot",
    "olmo3_mmlu_stem_5shot",
    "olmo3_humaneval_gold_bpb_3shot",
    "olmo3_mbpp_gold_bpb_3shot",
    "olmo3_math500_gold_bpb_0shot",
)

OLMO3_TASK_ALIASES = {
    "arc_challenge_test_bpb_5shot": "olmo3_arc_challenge_5shot",
    "arc_challenge_test_mc_5shot_fast": "olmo3_arc_challenge_5shot",
    "arc_easy_test_bpb_5shot": "olmo3_arc_easy_5shot",
    "arc_easy_test_mc_5shot_fast": "olmo3_arc_easy_5shot",
    "hellaswag_bpb_5shot": "olmo3_hellaswag_5shot",
    "mmlu_humanities_test_bpb_5shot": "olmo3_mmlu_humanities_5shot",
    "mmlu_humanities_test_mc_5shot_fast": "olmo3_mmlu_humanities_5shot",
    "mmlu_other_test_bpb_5shot": "olmo3_mmlu_other_5shot",
    "mmlu_other_test_mc_5shot_fast": "olmo3_mmlu_other_5shot",
    "mmlu_social_sciences_test_bpb_5shot": ("olmo3_mmlu_social_sciences_5shot"),
    "mmlu_social_sciences_test_mc_5shot_fast": ("olmo3_mmlu_social_sciences_5shot"),
    "mmlu_stem_test_bpb_5shot": "olmo3_mmlu_stem_5shot",
    "mmlu_stem_test_mc_5shot_fast": "olmo3_mmlu_stem_5shot",
    "codex_humaneval_gold_bpb_3shot": "olmo3_humaneval_gold_bpb_3shot",
    "codex_mbpp_gold_bpb_3shot": "olmo3_mbpp_gold_bpb_3shot",
    "minerva_math_500_gold_bpb_0shot": "olmo3_math500_gold_bpb_0shot",
}

SKIPPED_OLMO3_TASKS = {
    "basic_skills_arithmetic_rc_5shot": "No stock LM Harness task or dataset.",
    "basic_skills_coding_rc_5shot": "No stock LM Harness task or dataset.",
    "basic_skills_common_knowledge_rc_5shot": "No stock LM Harness task or dataset.",
    "basic_skills_logical_reasoning_rc_5shot": "No stock LM Harness task or dataset.",
    "basic_skills_pattern_rc_5shot": "No stock LM Harness task or dataset.",
    "basic_skills_string_operations_rc_5shot": "No stock LM Harness task or dataset.",
    "mt_mbpp_cpp_gold_bpb_3shot": "Stock Harness MBPP is Python only.",
    "mt_mbpp_java_gold_bpb_3shot": "Stock Harness MBPP is Python only.",
    "mt_mbpp_rust_gold_bpb_3shot": "Stock Harness MBPP is Python only.",
    "copycolors_10way_fast": "No stock LM Harness task or dataset.",
}

MMLU_CATEGORIES = {
    "stem": (
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ),
    "humanities": (
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ),
    "social_sciences": (
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ),
    "other": (
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ),
}

MC_METRICS = """\
metric_list:
  - metric: acc_raw
    aggregation: mean
    higher_is_better: true
  - metric: bpb_v1
    aggregation: bits_per_byte
    higher_is_better: false
  - metric: bpb_v2
    aggregation: bits_per_byte
    higher_is_better: false
"""

BPB_METRICS = """\
metric_list:
  - metric: bpb_v1
    aggregation: bits_per_byte
    higher_is_better: false
  - metric: bpb_v2
    aggregation: bits_per_byte
    higher_is_better: false
"""


def _write(path: Path, content: str) -> None:
    lines = content.lstrip("\n").splitlines()
    if not lines:
        path.write_text("")
        return

    base_indent = len(lines[0]) - len(lines[0].lstrip())
    normalized_lines = []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if line.strip() and indent >= base_indent:
            line = line[base_indent:]
        normalized_lines.append(line)
    path.write_text("\n".join(normalized_lines).rstrip() + "\n")


def _write_mc_tasks(output_dir: Path) -> None:
    _write(
        output_dir / "arc_challenge.yaml",
        f"""
        task: olmo3_arc_challenge_5shot
        dataset_path: allenai/ai2_arc
        dataset_name: ARC-Challenge
        output_type: multiple_choice
        training_split: train
        test_split: test
        fewshot_split: train
        process_docs: !function olmo3_harness_metrics.process_arc_docs
        doc_to_text: "{{{{_olmo3_prompt}}}}"
        doc_to_choice: "{{{{_olmo3_choices}}}}"
        doc_to_target: "{{{{_olmo3_gold}}}}"
        target_delimiter: " "
        num_fewshot: 5
        fewshot_config:
          sampler: default
        process_results: !function olmo3_harness_metrics.process_mc_results
        {MC_METRICS}
        """,
    )
    _write(
        output_dir / "arc_easy.yaml",
        (output_dir / "arc_challenge.yaml")
        .read_text()
        .replace("olmo3_arc_challenge_5shot", "olmo3_arc_easy_5shot")
        .replace("ARC-Challenge", "ARC-Easy"),
    )
    _write(
        output_dir / "hellaswag.yaml",
        f"""
        task: olmo3_hellaswag_5shot
        dataset_path: Rowan/hellaswag
        output_type: multiple_choice
        training_split: train
        validation_split: validation
        process_docs: !function olmo3_harness_metrics.process_hellaswag_docs
        doc_to_text: "{{{{_olmo3_prompt}}}}"
        doc_to_choice: "{{{{_olmo3_choices}}}}"
        doc_to_target: "{{{{_olmo3_gold}}}}"
        target_delimiter: " "
        num_fewshot: 5
        fewshot_config:
          sampler: default
        process_results: !function olmo3_harness_metrics.process_mc_results
        {MC_METRICS}
        """,
    )


def _write_mmlu_tasks(output_dir: Path) -> None:
    for category, subjects in MMLU_CATEGORIES.items():
        task_names = []
        for subject in subjects:
            task_name = f"olmo3_mmlu_{subject}_5shot"
            task_names.append(task_name)
            description = (
                "The following are multiple choice questions (with answers) about "
                f"{subject.replace('_', ' ')}.\n\n"
            )
            _write(
                output_dir / f"{task_name}.yaml",
                f"""
                task: {task_name}
                dataset_path: cais/mmlu
                dataset_name: {subject}
                output_type: multiple_choice
                test_split: test
                fewshot_split: dev
                process_docs: !function olmo3_harness_metrics.process_mmlu_docs
                description: {json.dumps(description)}
                doc_to_text: "{{{{_olmo3_prompt}}}}"
                doc_to_choice: "{{{{_olmo3_choices}}}}"
                doc_to_target: "{{{{_olmo3_gold}}}}"
                target_delimiter: " "
                num_fewshot: 5
                fewshot_config:
                  sampler: first_n
                process_results: !function olmo3_harness_metrics.process_mc_results
                {MC_METRICS}
                """,
            )

        task_lines = "\n".join(f"  - {task_name}" for task_name in task_names)
        _write(
            output_dir / f"olmo3_mmlu_{category}_group.yaml",
            (
                f"group: olmo3_mmlu_{category}_5shot\n"
                f"task:\n{task_lines}\n"
                "aggregate_metric_list:\n"
                "  - metric: acc_raw\n"
                "    weight_by_size: true\n"
                "  - metric: bpb_v1\n"
                "    weight_by_size: true\n"
                "  - metric: bpb_v2\n"
                "    weight_by_size: true\n"
            ),
        )


def _write_gold_bpb_tasks(output_dir: Path) -> None:
    _write(
        output_dir / "humaneval_gold_bpb.yaml",
        f"""
        task: olmo3_humaneval_gold_bpb_3shot
        dataset_path: openai/openai_humaneval
        output_type: loglikelihood
        test_split: test
        fewshot_split: test
        process_docs: !function olmo3_harness_metrics.process_humaneval_docs
        doc_to_text: "{{{{_olmo3_prompt}}}}"
        doc_to_target: "{{{{_olmo3_target}}}}"
        target_delimiter: ""
        num_fewshot: 3
        fewshot_config:
          sampler: default
        process_results: !function olmo3_harness_metrics.process_gold_bpb_results
        {BPB_METRICS}
        """,
    )
    _write(
        output_dir / "mbpp_gold_bpb.yaml",
        f"""
        task: olmo3_mbpp_gold_bpb_3shot
        dataset_path: google-research-datasets/mbpp
        dataset_name: full
        output_type: loglikelihood
        training_split: train
        test_split: test
        fewshot_split: train
        process_docs: !function olmo3_harness_metrics.process_mbpp_docs
        doc_to_text: "{{{{_olmo3_prompt}}}}"
        doc_to_target: "{{{{_olmo3_target}}}}"
        target_delimiter: ""
        num_fewshot: 3
        fewshot_config:
          sampler: default
        process_results: !function olmo3_harness_metrics.process_gold_bpb_results
        {BPB_METRICS}
        """,
    )
    _write(
        output_dir / "math500_gold_bpb.yaml",
        f"""
        task: olmo3_math500_gold_bpb_0shot
        dataset_path: HuggingFaceH4/MATH-500
        dataset_name: default
        output_type: loglikelihood
        test_split: test
        process_docs: !function olmo3_harness_metrics.process_math500_docs
        doc_to_text: "{{{{_olmo3_prompt}}}}"
        doc_to_target: "{{{{_olmo3_target}}}}"
        target_delimiter: ""
        num_fewshot: 0
        process_results: !function olmo3_harness_metrics.process_gold_bpb_results
        {BPB_METRICS}
        """,
    )


def prepare_task_configs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_source = Path(__file__).with_name("olmo3_harness_metrics.py")
    shutil.copy2(metrics_source, output_dir / metrics_source.name)
    _write_mc_tasks(output_dir)
    _write_mmlu_tasks(output_dir)
    _write_gold_bpb_tasks(output_dir)
    return output_dir
