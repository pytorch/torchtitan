# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest

from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_autotune import (
    _full_sm100_configs,
    _initial_configs,
    _microbenchmark_command,
    _search_configs,
    INITIAL_CONFIGS,
    INITIAL_CONFIGS_16B,
    PRIORITY_CONFIGS,
)
from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench import (
    _kernel_options,
    CASES,
)
from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b import (
    CASES as CASES_16B,
)


class TestCodaFusionMicrobench(unittest.TestCase):
    def test_case_inventory(self) -> None:
        self.assertEqual(len(CASES), 12)
        self.assertEqual(
            {case.pattern for case in CASES.values()},
            {
                "B1",
                "B2",
                "B4",
                "B5",
                "B6",
                "B7",
                "F2-Q",
                "F2-KV",
                "F3-A",
                "F3-B",
                "F4",
                "F6",
            },
        )

    def test_tuning_and_fast_math_defaults(self) -> None:
        case = CASES["f4_shared_expert_swiglu"]
        options = _kernel_options(
            case,
            configs=(' {"tile_m": 128, "cluster_n": 1}',),
        )
        self.assertEqual(
            options,
            (
                {
                    "backend": "QUACK",
                    "tuned": True,
                    "fast_math": True,
                    "config": {"tile_m": 128, "cluster_n": 1},
                },
                {
                    "backend": "QUACK",
                    "tuned": True,
                    "config": {"tile_m": 128, "cluster_n": 1},
                },
            ),
        )

    def test_16b_case_inventory(self) -> None:
        self.assertEqual(len(CASES_16B), 13)
        self.assertEqual(
            {case.pattern for case in CASES_16B.values()},
            {"B1", "B2", "B4", "B5", "B6", "B7", "F2-KV", "F3-A", "F3-B", "F4"},
        )
        self.assertNotIn("f2_q_rmsnorm", CASES_16B)
        self.assertNotIn("f6_router_sigmoid_bias", CASES_16B)
        self.assertEqual(
            CASES_16B["b1_lm_head_input_grad_cast"].shape,
            "(2048, 102400) @ (102400, 2048); 8 chunks per step",
        )

    def test_every_case_always_uses_tuned_mode(self) -> None:
        for case in CASES.values():
            with self.subTest(case=case.name):
                options = _kernel_options(case, configs=())
                self.assertTrue(all(option["tuned"] for option in options))
                self.assertEqual(
                    tuple(
                        index
                        for index, option in enumerate(options)
                        if option.get("fast_math")
                    ),
                    case.fast_math_flex_gemms,
                )

    def test_config_count_must_match_flex_gemm_count(self) -> None:
        case = CASES["f2_q_rmsnorm"]
        with self.assertRaisesRegex(ValueError, "zero, one, or 2"):
            _kernel_options(
                case,
                configs=("{}", "{}", "{}"),
            )

    def test_autotune_search_spaces(self) -> None:
        self.assertEqual(len(PRIORITY_CONFIGS), 12)
        self.assertEqual(len(_full_sm100_configs()), 74)
        self.assertEqual(len(_search_configs("full", None)), 74)
        for config in _full_sm100_configs():
            self.assertFalse(config["swap_ab"])
            self.assertFalse(config["use_tma_gather"])

    def test_autotune_initial_configs_cover_every_case(self) -> None:
        self.assertEqual(INITIAL_CONFIGS.keys(), CASES.keys())
        for name, case in CASES.items():
            args = argparse.Namespace(case=name, base_config=[])
            self.assertEqual(len(_initial_configs(args)), case.num_flex_gemms)

        self.assertEqual(INITIAL_CONFIGS_16B.keys(), CASES_16B.keys())
        for name, case in CASES_16B.items():
            args = argparse.Namespace(
                suite="16b",
                case=name,
                base_config=[],
            )
            self.assertEqual(len(_initial_configs(args)), case.num_flex_gemms)

    def test_autotune_uses_packaged_microbenchmark(self) -> None:
        self.assertEqual(
            _microbenchmark_command()[-1],
            "torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench",
        )
        self.assertEqual(
            _microbenchmark_command("16b")[-1],
            "torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b",
        )


if __name__ == "__main__":
    unittest.main()
