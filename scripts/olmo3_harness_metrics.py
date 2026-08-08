# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


def _with_bpb_metrics(loglikelihood: float, continuation: str) -> dict[str, Any]:
    without_leading_space = (
        continuation[1:] if continuation.startswith(" ") else continuation
    )
    return {
        "bpb_v1": (loglikelihood, len(without_leading_space.encode("utf-8"))),
        "bpb_v2": (loglikelihood, len(continuation.encode("utf-8"))),
    }


def process_mc_results(
    doc: dict[str, Any], results: list[tuple[float, bool]]
) -> dict[str, Any]:
    loglikelihoods = [result[0] for result in results]
    gold = int(doc["_olmo3_gold"])
    prediction = max(range(len(loglikelihoods)), key=loglikelihoods.__getitem__)
    continuation = " " + doc["_olmo3_choices"][gold]
    return {
        "acc_raw": float(prediction == gold),
        **_with_bpb_metrics(loglikelihoods[gold], continuation),
    }


def process_gold_bpb_results(
    doc: dict[str, Any], results: list[tuple[float, bool]]
) -> dict[str, Any]:
    return _with_bpb_metrics(results[0][0], doc["_olmo3_target"])


def process_arc_docs(dataset):
    def process_doc(doc):
        labels = [str(label) for label in doc["choices"]["label"]]
        answer_key = str(doc["answerKey"])
        if answer_key in labels:
            gold = labels.index(answer_key)
        else:
            gold = int(answer_key) - 1
        choices = [chr(ord("A") + i) for i in range(len(labels))]
        rendered_choices = "\n".join(
            f" {label}. {text}"
            for label, text in zip(choices, doc["choices"]["text"], strict=True)
        )
        return {
            "_olmo3_prompt": f"Question: {doc['question']}\n{rendered_choices}\nAnswer:",
            "_olmo3_choices": choices,
            "_olmo3_gold": gold,
        }

    return dataset.map(process_doc)


def process_hellaswag_docs(dataset):
    from lm_eval.tasks.hellaswag.utils import process_docs

    dataset = process_docs(dataset)

    def process_doc(doc):
        return {
            "_olmo3_prompt": doc["query"],
            "_olmo3_choices": doc["choices"],
            "_olmo3_gold": int(doc["gold"]),
        }

    return dataset.map(process_doc)


def process_mmlu_docs(dataset):
    def process_doc(doc):
        labels = ["A", "B", "C", "D"]
        rendered_choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(labels, doc["choices"], strict=True)
        )
        return {
            "_olmo3_prompt": (
                f"{doc['question'].strip()}\n{rendered_choices}\nAnswer:"
            ),
            "_olmo3_choices": labels,
            "_olmo3_gold": int(doc["answer"]),
        }

    return dataset.map(process_doc)


def process_humaneval_docs(dataset):
    return dataset.map(
        lambda doc: {
            "_olmo3_prompt": doc["prompt"],
            "_olmo3_target": doc["canonical_solution"],
        }
    )


def process_mbpp_docs(dataset):
    return dataset.map(
        lambda doc: {
            "_olmo3_prompt": (
                "You are an expert Python programmer. Complete this task:\n"
                f"{doc['text']}\n"
            ),
            "_olmo3_target": doc["code"],
        }
    )


def process_math500_docs(dataset):
    return dataset.map(
        lambda doc: {
            "_olmo3_prompt": f"Problem: {doc['problem']}\nAnswer:",
            "_olmo3_target": doc["solution"],
        }
    )
