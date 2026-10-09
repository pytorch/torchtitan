# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DAPO-Math dataset, environment, and rubric."""

from __future__ import annotations

import asyncio
import time

import pytest
from datasets import Dataset

from torchtitan.rl.examples.dapo_math import (
    AIME2025Dataset,
    DapoMathDataset,
    DapoMathEnv,
    DapoMathSample,
    data as math_data,
    grader,
    MathVerifyPool,
    RewardMathVerify,
    score_math_response,
)
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.types import RolloutTurnID


def _dapo_rows() -> list[dict]:
    return [
        {
            "source_prompt": [{"role": "user", "content": "problem 1"}],
            "prompt": "problem 1",
            "ground_truth": "34",
        },
        {
            "source_prompt": [{"role": "user", "content": "problem 2"}],
            "prompt": "problem 2",
            "ground_truth": "113",
        },
        {
            "source_prompt": [{"role": "user", "content": "problem 3"}],
            "prompt": "problem 3",
            "ground_truth": "7",
        },
    ]


def test_dapo_dataset_is_deterministic_and_resumable(monkeypatch) -> None:
    monkeypatch.setattr(math_data, "load_dataset", lambda *args, **kwargs: _dapo_rows())
    config = DapoMathDataset.Config(seed=7)
    first = config.build()
    second = config.build()
    assert [next(first) for _ in range(3)] == [next(second) for _ in range(3)]

    checkpoint = first.state_dict()
    expected = [next(first) for _ in range(3)]
    resumed = config.build()
    resumed.load_state_dict(checkpoint)
    assert [next(resumed) for _ in range(3)] == expected
    assert all(r"Answer: \boxed{" in sample.prompt for sample in expected)


def test_aime_dataset_combines_both_subsets(monkeypatch) -> None:
    def load_dataset(repo_id, subset, *, split):
        del repo_id, split
        answer = r"42^\circ" if subset == "AIME2025-I" else r"\boxed{42}"
        return Dataset.from_list([{"question": f"{subset} question", "answer": answer}])

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = AIME2025Dataset.Config(num_samples=2).build()
    samples = [next(dataset), next(dataset)]
    assert [sample.ground_truth for sample in samples] == [r"42^\circ", r"\boxed{42}"]
    assert "AIME2025-I question" in samples[0].prompt
    assert "AIME2025-II question" in samples[1].prompt
    assert all(r"Answer: \boxed{" in sample.prompt for sample in samples)


def test_aime_dataset_restarts_after_configured_num_samples(monkeypatch) -> None:
    def load_dataset(repo_id, subset, *, split):
        del repo_id, split
        return Dataset.from_list([{"question": f"{subset} question", "answer": "42"}])

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = AIME2025Dataset.Config(num_samples=1).build()
    first = next(dataset)
    assert next(dataset) == first


def test_env_is_single_turn() -> None:
    env = DapoMathEnv.Config().build(
        env_input=DapoMathSample(prompt="solve me", ground_truth="3"),
    )
    initial = asyncio.run(env.init())
    assert initial.init_prompt_messages == [{"role": "user", "content": "solve me"}]
    assert asyncio.run(env.step({"role": "assistant", "content": "Answer: 3"})).done


def _rollout(response: str) -> Rollout:
    return Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=[
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[-0.1],
                completion_message={"role": "assistant", "content": response},
            )
        ],
    )


def test_math_verifier_requires_a_boxed_answer() -> None:
    assert score_math_response(r"work\nAnswer: \boxed{34}", "34") == 1.0
    assert score_math_response(r"work\n\boxed{\frac{68}{2}}", "34") == 1.0
    assert score_math_response("work\nAnswer: $34$", "34") == 0.0
    assert score_math_response("work\nAnswer: 34", "34") == 0.0
    assert score_math_response("work mentions 34", "34") == 0.0


def test_math_verifier_parses_non_integer_gold_answers() -> None:
    assert score_math_response(r"\boxed{2\sqrt{3}}", r"2\sqrt{3}") == 1.0
    assert score_math_response(r"\boxed{2}", r"2\sqrt{3}") == 0.0
    assert score_math_response(r"\boxed{(1,2)}", "(1,2)") == 1.0
    assert score_math_response(r"\boxed{\frac{\pi}{4}}", r"\pi/4") == 1.0


def test_math_verifier_reads_thin_space_digit_groups_as_one_number() -> None:
    # Math-Verify alone reads `10\,002` as 10 + 002 = 12.
    assert score_math_response(r"\boxed{10\,002}", "10002") == 1.0
    assert score_math_response(r"\boxed{1\,091\,328}", "1091328") == 1.0
    assert score_math_response(r"\boxed{50\,413.5}", "50413.5") == 1.0
    assert score_math_response(r"\boxed{10002}", r"10\,002") == 1.0
    assert score_math_response(r"\boxed{10\,002}", "12") == 0.0
    # Only a group of exactly 3 digits joins the number before it.
    assert score_math_response(r"\boxed{2\,3}", "23") == 0.0


def test_math_verifier_compares_word_answers_as_strings() -> None:
    assert score_math_response(r"\boxed{Indonesian}", "indonesian") == 1.0
    assert score_math_response(r"\boxed{eat}", "tea") == 0.0
    # A real padded gold. Spaces between letters don't hide a wrong order.
    assert score_math_response(r"\boxed{EDACB}", " EDACB ") == 1.0
    assert score_math_response(r"\boxed{DAECB}", " EDACB ") == 0.0
    assert score_math_response(r"\boxed{D A E C B}", " EDACB ") == 0.0
    assert score_math_response(r"\boxed{D\,A\,E\,C\,B}", " EDACB ") == 0.0
    # Wrapped letters reach Math-Verify, which reads the gold as a product.
    for wrapped in [
        r"(DAECB)",
        r"\text{DAECB}",
        r"D\cdot A\cdot E\cdot C\cdot B",
        r"D\quad A\quad E\quad C\quad B",
    ]:
        assert score_math_response(rf"\boxed{{{wrapped}}}", " EDACB ") == 0.0
    assert score_math_response(r"\boxed{(EDACB)}", " EDACB ") == 1.0
    # One- and two-letter answers stay math: `ba` is `ab`.
    assert score_math_response(r"\boxed{ba}", "ab") == 1.0


def test_math_verifier_scores_the_right_side_of_a_named_answer() -> None:
    # Math-Verify alone parses `A_{\min}= \frac12` to nothing and `k_{\max}(n)=...` to 2.
    assert score_math_response(r"\boxed{A_{\min}= \frac12}", r"\frac{1}{2}") == 1.0
    response = r"\boxed{\,k_{\max}(n)=2^{\,n-1}-n\,}"
    assert score_math_response(response, "2^{n-1} - n") == 1.0
    assert score_math_response(r"\boxed{A_{\min}= \frac13}", r"\frac{1}{2}") == 0.0
    # Not a `name = value`: a list, an `=` inside braces, or `>=`.
    assert score_math_response(r"\boxed{x = 2 \text{ or } x = 3}", "3") == 0.0
    assert score_math_response(r"\boxed{x = 2 or x = 3}", "3") == 0.0
    assert score_math_response(r"\boxed{a=4,\ b=-1}", "-1") == 0.0
    assert score_math_response(r"\boxed{\sum_{k=1}^{n} k}", "1") == 0.0
    assert score_math_response(r"\boxed{x >= 3}", "3") == 0.0
    # A gold equation keeps its left side: Math-Verify drops it when the answer has none.
    assert score_math_response(r"\boxed{y = 3}", "x = 3") == 0.0
    assert score_math_response(r"\boxed{f(f(x)) = x}", "f(x) = x") == 0.0


def test_math_verifier_reads_degrees_as_radians_against_pi() -> None:
    assert score_math_response(r"\boxed{60^\circ}", r"\frac{\pi}{3}") == 1.0
    assert score_math_response(r"\boxed{\frac{\pi}{3}}", r"60^{\circ}") == 1.0
    assert score_math_response(r"\boxed{90^\circ-30^\circ}", r"\frac{\pi}{3}") == 1.0
    assert score_math_response(r"\boxed{45^\circ}", r"\frac{\pi}{3}") == 0.0
    # Only the side without a `\pi` is rewritten; without a `\pi`, degrees stay as written.
    assert score_math_response(r"\boxed{\frac{\pi}{3} = 60^\circ}", "60") == 1.0
    assert score_math_response(r"\boxed{60}", r"60^\circ") == 1.0


def test_math_verifier_reads_leqq_as_le() -> None:
    # Math-Verify alone reads `p\leqq0` as `p \le q \cdot 0`.
    assert score_math_response(r"\boxed{p \le 0}", r"p\leqq0") == 1.0
    assert score_math_response(r"\boxed{p \leqq 0}", r"p \le 0") == 1.0
    assert score_math_response(r"\boxed{[1, 2]}", r"1\leqqx\leqq2") == 1.0
    assert score_math_response(r"\boxed{p \ge 0}", r"p\leqq0") == 0.0


def test_math_verifier_drops_delimiter_sizes() -> None:
    # Math-Verify alone parses `\bigl(` and `\Big|` to nothing, and `\biggl[0, 1\biggr)` as 1.
    response = r"\boxed{4\pi\bigl(2-\sqrt3\bigr)}"
    assert score_math_response(response, r"4 \pi (2 - \sqrt{3})") == 1.0
    assert score_math_response(response, r"4 \pi (2 + \sqrt{3})") == 0.0
    assert score_math_response(r"\boxed{\Big|x\Big|}", "|x|") == 1.0
    assert score_math_response(r"\boxed{|x|}", r"\Big|x\Big|") == 1.0
    assert score_math_response(r"\boxed{\biggl[0, 1\biggr)}", "[0, 1)") == 1.0
    assert score_math_response(r"\boxed{\biggl[0, 1\biggr)}", "1") == 0.0
    # A JSON-escaped `\\bigl(` stays: dropping `\bigl` would leave `\(`, which starts inline math.
    assert score_math_response(r"\boxed{2\\bigl(3\\bigr)}", "3") == 0.0


def test_math_verifier_drops_a_percent_sign_on_one_side() -> None:
    # Math-Verify alone matches `12\%` with `12`, but not these.
    assert score_math_response(r"\boxed{-12\%}", "-12") == 1.0
    assert score_math_response(r"\boxed{11.54\%}", "11.54") == 1.0
    assert score_math_response(r"\boxed{-13\%}", "-12") == 0.0
    # `12\%` still matches `0.12`.
    assert score_math_response(r"\boxed{12\%}", "0.12") == 1.0


def test_math_verifier_retries_without_a_trailing_qualifier() -> None:
    gold = r"\frac{3\pi}{2}+2k\pi"
    response = r"\boxed{x = \frac{3\pi}{2} + 2\pi k,\ k \in \mathbb{Z}}"
    assert score_math_response(response, gold) == 1.0
    assert score_math_response(r"\boxed{2^n \text{ for all } n}", "2^n") == 1.0
    response = r"\boxed{x_n=n\ \ \text{for all}\ n\in\mathbb N}"
    assert score_math_response(response, "x_{n} = n") == 1.0
    response = r"\boxed{(1,k)\ \forall k\in\mathbb{Z}^+}"
    assert score_math_response(response, "(1, k)") == 1.0
    response = r"\boxed{x = \frac{\pi}{2} + 2\pi k,\ k \in \mathbb{Z}}"
    assert score_math_response(response, gold) == 0.0
    # A piecewise answer keeps its other cases: 2 for n >= 3 and 1 below is not 2.
    response = r"\boxed{2 \text{ for all } n \ge 3,\ 1 \text{ for } n = 2}"
    assert score_math_response(response, "2") == 0.0
    assert score_math_response(r"\boxed{2 \forall n \geq 3,\ 1\ (n < 3)}", "2") == 0.0
    assert (
        score_math_response(r"\boxed{2 \forall n \geq 3 \quad 1\ (n < 3)}", "2") == 0.0
    )
    response = r"\boxed{\begin{cases} 1 & \forall n \geq 2 \\ 0 & n < 2 \end{cases}}"
    assert score_math_response(response, "1") == 0.0


def test_math_verifier_matches_a_solved_inequality_with_its_interval() -> None:
    gold = r"(-1, \sqrt{2}-1)"
    assert score_math_response(r"\boxed{-1 < x < \sqrt{2}-1}", gold) == 1.0
    assert score_math_response(r"\boxed{a \le 10}", r"(-\infty,10]") == 1.0
    # Open and closed ends must match.
    assert score_math_response(r"\boxed{-1 \le x < \sqrt{2}-1}", gold) == 0.0
    assert score_math_response(r"\boxed{a < 10}", r"(-\infty,10]") == 0.0
    # An unsolved inequality, e.g. copied from the problem, is not its solution set.
    assert score_math_response(r"\boxed{x^2 - 3x + 2 < 0}", "(1, 2)") == 0.0
    # The roots of the equation are not the solution set of the inequality.
    assert score_math_response(r"\boxed{1 < x < 2}", "1, 2") == 0.0
    assert score_math_response(r"\boxed{1 < x < 2}", "x = 1, x = 2") == 0.0


def test_math_verifier_drops_a_gold_left_side_without_symbols() -> None:
    # Math-Verify drops the left side of `x = 5`, but not of these.
    gold = r"\gamma = 120^\circ"
    assert score_math_response(r"\boxed{120^\circ}", gold) == 1.0
    assert score_math_response(r"\boxed{1}", r"\frac{9}{9} = 1") == 1.0
    assert score_math_response(r"\boxed{\frac{25}{2}}", "f(80) = 12.5") == 1.0
    assert score_math_response(r"\boxed{2}", r"\frac{9}{9} = 1") == 0.0
    # The left side stays when it has a symbol, or when the answer has its own.
    assert score_math_response(r"\boxed{7}", "2x + 1 = 7") == 0.0
    assert score_math_response(r"\boxed{\beta = 120^\circ}", gold) == 0.0
    # Only an equation's left side drops, not an inequality's.
    assert score_math_response(r"\boxed{a}", "0 <= a") == 0.0
    # Not at an `=` in braces.
    assert score_math_response(r"\boxed{1}", r"55 = \sum_{k=1}^{10} k") == 0.0


def test_math_verifier_compares_at_the_precision_of_a_rounded_gold() -> None:
    assert score_math_response(r"\boxed{\frac{50}{3}}", "16.67") == 1.0
    assert score_math_response(r"\boxed{\frac{963}{160}}", "6.02") == 1.0
    assert score_math_response(r"\boxed{\frac{250}{9}}", r"27.78 \%") == 1.0
    assert score_math_response(r"\boxed{\frac{50}{3}}", "16.66") == 0.0
    assert score_math_response(r"\boxed{16.7}", "16.67") == 0.0
    # Under 3 significant digits the gold may be exact: 19/100 is not 0.2.
    assert score_math_response(r"\boxed{\frac{19}{100}}", "0.2") == 0.0
    # A percent gold rounds in percent: 27.5% is not 27.78%.
    assert score_math_response(r"\boxed{27.5\%}", r"27.78\%") == 0.0
    # More than 6 decimals compare at 6, Math-Verify's default.
    assert score_math_response(r"\boxed{0.31831}", "0.3183098") == 1.0


def test_math_verifier_drops_units_but_not_variables() -> None:
    # Math-Verify's units include these letters, so it reads `2 - c` as `2 -`.
    for letter in "cmgthsdlo":
        response = r"\boxed{-c + 2}".replace("c", letter)
        assert score_math_response(response, "2 - " + letter) == 1.0
    assert score_math_response(r"\boxed{2 - c}", "2-c") == 1.0
    assert score_math_response(r"\boxed{3x + a + b + c}", "a + b + c + 3x") == 1.0
    assert score_math_response(r"\boxed{\frac{8h}{5}}", r"\frac{8}{5} h") == 1.0
    assert score_math_response(r"\boxed{100a+b+c}", " 100a + b + c ") == 1.0
    # A unit after a number, or in `\text`, still drops.
    assert score_math_response(r"\boxed{5 cm}", "5") == 1.0
    assert score_math_response(r"\boxed{3 h}", "3") == 1.0
    assert score_math_response(r"\boxed{1}", r"1 \text{ cm}") == 1.0
    assert score_math_response(r"\boxed{EF = 8}", r"EF = 8 \text{ cm}") == 1.0
    assert score_math_response(r"\boxed{2^{n-1} \text{ ways}}", "2^{n-1}") == 1.0
    # If keeping the units does not parse, Math-Verify's parse is used.
    response = r"\boxed{f(x) = a x + b \text{ where } a, b \in \mathbb{R}}"
    assert score_math_response(response, "f(x)=ax+b") == 1.0
    # A dropped variable no longer matches what is left.
    assert score_math_response(r"\boxed{a + c}", "a + b + c") == 0.0
    assert score_math_response(r"\boxed{2 - m}", "2 - c") == 0.0
    assert score_math_response(r"\boxed{4}", "n = 4m") == 0.0
    assert score_math_response(r"\boxed{70}", "15+55m") == 0.0
    assert score_math_response(r"\boxed{\frac{2}{3} R^2}", r"\frac{2}{3} R^2 h") == 0.0
    # Math-Verify reads the padded gold as 100, and `12 + 3 h (` as 3.
    assert score_math_response(r"\boxed{100}", " 100a + b + c ") == 0.0
    assert score_math_response(r"\boxed{12 + 3 h (}", "3") == 0.0


def test_math_verifier_uses_the_last_boxed_answer() -> None:
    response = r"Work: \boxed{2003^{2002^{2001}}}" "\n" r"Answer: \boxed{34}"
    assert score_math_response(response, "34") == 1.0


def test_math_verifier_rejects_unboxed_large_intermediate_expression() -> None:
    response = r"Work: \[2003^{2002^{2001}}\]" "\n" r"Final: \[Answer: 009\]"
    assert score_math_response(response, "241") == 0.0


def test_math_verify_pool_matches_in_process_scores() -> None:
    pool = MathVerifyPool(num_processes=2, timeout_seconds=5.0)
    cases = [
        (r"work\nAnswer: \boxed{34}", "34"),
        (r"work\n\boxed{\frac{68}{2}}", "34"),
        (r"work\nAnswer: \boxed{35}", "34"),
        ("work\nAnswer: 34", "34"),
        (r"\boxed{2\sqrt{3}}", r"2\sqrt{3}"),
    ]

    async def score_all() -> list[float]:
        return await asyncio.gather(*(pool.score(*case) for case in cases))

    assert asyncio.run(score_all()) == [score_math_response(*case) for case in cases]


def test_reward_scores_a_hung_answer_zero_while_other_answers_finish() -> None:
    # sympy computes 2000**(2000**2000) in one C call that does not finish.
    reward = RewardMathVerify.Config(timeout_seconds=2.0, num_processes=2).build()
    sample = DapoMathSample(prompt="problem", ground_truth="34")
    finish_seconds = {}

    async def score_and_time(name: str, response: str) -> float:
        reward_value = await reward(_rollout(response), sample)
        finish_seconds[name] = time.monotonic() - start
        return reward_value

    async def score_all() -> list[float]:
        hung = score_and_time("hung", r"work\nAnswer: \boxed{2000^{2000^{2000}}}")
        normal = [
            score_and_time(f"normal_{i}", r"work\nAnswer: \boxed{34}") for i in range(4)
        ]
        return await asyncio.gather(hung, *normal)

    start = time.monotonic()
    assert asyncio.run(score_all()) == [0.0, 1.0, 1.0, 1.0, 1.0]
    # The 2 s timeout, plus up to 5 s to start the grader process.
    assert finish_seconds["hung"] < 2.0 + 5.0
    assert max(finish_seconds[f"normal_{i}"] for i in range(4)) < finish_seconds["hung"]


def test_math_verify_pool_replaces_a_killed_process() -> None:
    # One process, so the second score runs on its replacement.
    pool = MathVerifyPool(num_processes=1, timeout_seconds=1.0)
    assert asyncio.run(pool.score(r"\boxed{2000^{2000^{2000}}}", "34")) == 0.0
    assert asyncio.run(pool.score(r"\boxed{34}", "34")) == 1.0


def test_math_verify_pool_raises_when_a_process_fails_to_start(
    monkeypatch, tmp_path
) -> None:
    # A process that prints something other than "ready" and keeps running.
    script = tmp_path / "noisy.py"
    script.write_text("import sys\nprint('noise', flush=True)\nsys.stdin.read()\n")
    monkeypatch.setattr(grader, "__file__", str(script))
    pool = MathVerifyPool(num_processes=1, timeout_seconds=1.0)
    with pytest.raises(RuntimeError, match="noise"):
        asyncio.run(pool.score(r"\boxed{34}", "34"))


def test_reward_handles_equivalent_latex_and_units() -> None:
    reward = RewardMathVerify.Config().build()
    sample = DapoMathSample(prompt="problem", ground_truth=r"336^\circ")
    assert asyncio.run(reward(_rollout(r"work\nAnswer: \boxed{336}"), sample)) == 1.0
    assert asyncio.run(reward(_rollout(r"work\nAnswer: \boxed{335}"), sample)) == 0.0
