# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../torchtitan/distributed/production_debt.py",
)
spec = importlib.util.spec_from_file_location("torchtitan_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["torchtitan_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtTitanGate = production_debt_mod.ProductionDebtTitanGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtTitanGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtTitanGate(
            never_equate_intent_to_approval=True,
            max_acceptable_tdi=12.0,
        )

    def test_clean_training_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_training_step(
            run_id="torchtitan_llama3_1_405b_fsdp2_tp_cp_run",
            allocated_pipeline_bytes=65000000000,
            peak_bubble_memory_bytes=68000000000,
            step_latency_seconds=0.36,
            nccl_comm_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.tdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_training_step_fails_debt(self) -> None:
        report = self.gate.evaluate_training_step(
            run_id="uncalibrated_3d_parallel_run",
            allocated_pipeline_bytes=65000000000,
            peak_bubble_memory_bytes=180000000000,  # 2.77x bubble sprawl
            step_latency_seconds=3.2,  # High step latency
            nccl_comm_stalls=3,  # 3 NCCL stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.tdi_score, 50.0)
        self.assertIn("HIGH_3D_PIPELINE_BUBBLE_SPRAWL_2.77X", report.critical_smells)
        self.assertIn("HIGH_DISTRIBUTED_STEP_LATENCY_3.20S", report.critical_smells)
        self.assertIn("DETECTED_3_NCCL_COLLECTIVE_COMM_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_DCP_CHECKPOINT_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_training_step("run-1")
        self.gate.evaluate_training_step("run-2")
        self.gate.evaluate_training_step("run-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
