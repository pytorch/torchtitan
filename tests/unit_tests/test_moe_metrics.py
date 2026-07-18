# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from torchtitan.components.moe_metrics import (
    collect_dense_gemm_templates,
    GroupedGemmRecord,
    maybe_record_grouped_gemm,
    MoEMetricCollector,
    MoEMetricsConfig,
    set_active_moe_metric_collector,
)


def _make_record(step: int = 1) -> GroupedGemmRecord:
    return GroupedGemmRecord(
        step=step,
        layer_id=0,
        micro_batch_id=0,
        rank=0,
        ep_rank=0,
        ep_size=1,
        num_local_experts=2,
        top_k=6,
        tokens_per_local_expert=(3, 5),
        padded_tokens_per_local_expert=(4, 8),
        gemm_w1=(12, 16, 32),
        gemm_w3=(12, 16, 32),
        gemm_w2=(12, 32, 16),
        dtype="bfloat16",
        dispatcher="alltoall",
    )


class TestMoEMetricCollector(unittest.TestCase):
    def test_disabled_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=False),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )
            collector.begin_step(1)
            collector.record(_make_record())
            collector.flush()
            collector.close()

            self.assertFalse(collector.is_enabled())
            self.assertFalse((Path(tmpdir) / "moe_metrics").exists())

    def test_rank_filter_rank0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MoEMetricsConfig(enabled=True, ranks="rank0")
            collector_rank1 = MoEMetricCollector(
                config=cfg,
                dump_folder=tmpdir,
                rank=1,
                world_size=8,
            )
            self.assertFalse(collector_rank1.is_enabled())
            collector_rank1.close()

            collector_rank0 = MoEMetricCollector(
                config=cfg,
                dump_folder=tmpdir,
                rank=0,
                world_size=8,
            )
            self.assertTrue(collector_rank0.is_enabled())
            collector_rank0.close()

    def test_sample_every_gates_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=True, sample_every=2, sinks=["jsonl"]),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )

            collector.begin_step(1)
            collector.record(_make_record(step=1))
            collector.flush()

            collector.begin_step(2)
            collector.record(_make_record(step=2))
            collector.flush()
            collector.close()

            out_file = Path(tmpdir) / "moe_metrics" / "rank_0.jsonl"
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('"step": 2', lines[0])

    def test_grouped_gemm_hook_records_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=True, sample_every=1, sinks=["jsonl"]),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )
            collector.begin_step(3)
            set_active_moe_metric_collector(collector)

            x_RD = torch.randn(12, 16)
            w1_EFD = torch.randn(2, 32, 16)
            w2_EDF = torch.randn(2, 16, 32)
            w3_EFD = torch.randn(2, 32, 16)
            num_tokens_per_expert_E = torch.tensor([4, 8], dtype=torch.int64)
            padded_num_tokens_per_expert_E = torch.tensor([8, 8], dtype=torch.int64)

            dispatch_metadata = SimpleNamespace(
                num_tokens_per_local_expert_e=num_tokens_per_expert_E,
                padded_num_tokens_per_local_expert_e=padded_num_tokens_per_expert_E,
                dispatcher="alltoall",
            )
            maybe_record_grouped_gemm(
                x_RD=x_RD,
                w1_EFD=w1_EFD,
                w2_EDF=w2_EDF,
                w3_EFD=w3_EFD,
                num_tokens_per_expert_E=num_tokens_per_expert_E,
                dispatch_metadata=dispatch_metadata,
                layer_id=7,
                micro_batch_id=1,
                top_k=6,
            )
            collector.flush()
            collector.close()
            set_active_moe_metric_collector(None)

            out_file = Path(tmpdir) / "moe_metrics" / "rank_0.jsonl"
            self.assertTrue(out_file.exists())
            line = out_file.read_text(encoding="utf-8").strip()
            self.assertIn('"step": 3', line)
            self.assertIn('"layer_id": 7', line)
            self.assertIn('"micro_batch_id": 1', line)
            self.assertIn('"top_k": 6', line)
            self.assertIn('"gemm_w1": [12, 16, 32]', line)
            self.assertIn('"gemm_w3": [12, 16, 32]', line)
            self.assertIn('"gemm_w2": [12, 32, 16]', line)
            self.assertIn('"tokens_per_local_expert": [4, 8]', line)
            self.assertIn('"padded_tokens_per_local_expert": [8, 8]', line)
            self.assertIn('"dispatcher": "alltoall"', line)

    def test_grouped_gemm_hook_disabled_collector_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=False),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )
            collector.begin_step(1)
            set_active_moe_metric_collector(collector)

            maybe_record_grouped_gemm(
                x_RD=torch.randn(2, 4),
                w1_EFD=torch.randn(1, 8, 4),
                w2_EDF=torch.randn(1, 4, 8),
                w3_EFD=torch.randn(1, 8, 4),
                num_tokens_per_expert_E=torch.tensor([2], dtype=torch.int64),
            )
            collector.flush()
            collector.close()
            set_active_moe_metric_collector(None)

            self.assertFalse((Path(tmpdir) / "moe_metrics").exists())

    def test_hook_compiles_fullgraph_when_no_collector(self):
        """The disabled path (no collector installed) must be compile-safe.

        ``GroupedExperts._experts_forward`` is compiled with ``fullgraph=True``,
        and the hook runs inside it. With no collector installed, the module
        global ``_COLLECTOR_INSTALLED`` is ``False``, so Dynamo specializes on
        it and dead-code-eliminates the hook body instead of choking on the
        ``ContextVar`` read. This guards the common metrics-off + compile-on
        configuration against regressions.
        """

        set_active_moe_metric_collector(None)

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1_EFD = torch.nn.Parameter(torch.randn(2, 8, 4))
                self.w2_EDF = torch.nn.Parameter(torch.randn(2, 4, 8))
                self.w3_EFD = torch.nn.Parameter(torch.randn(2, 8, 4))

            def forward(self, x_RD, counts):
                maybe_record_grouped_gemm(
                    x_RD=x_RD,
                    w1_EFD=self.w1_EFD,
                    w2_EDF=self.w2_EDF,
                    w3_EFD=self.w3_EFD,
                    num_tokens_per_expert_E=counts,
                )
                return x_RD @ self.w1_EFD[0].t()

        torch._dynamo.reset()
        mod = Tiny()
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(3, 4)
        counts = torch.tensor([1, 2], dtype=torch.int64)
        # Must not raise Unsupported / graph break under fullgraph.
        out = compiled(x, counts)
        self.assertEqual(out.shape, (3, 8))


class TestHistogramSinkAndManifest(unittest.TestCase):
    def test_histogram_accumulates_per_layer_token_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=True, sinks=["histogram"]),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )
            # Two steps: layer 0 sees token counts (3, 5) and (3, 7).
            collector.begin_step(1)
            collector.record(
                GroupedGemmRecord(
                    step=1,
                    layer_id=0,
                    micro_batch_id=0,
                    rank=0,
                    ep_rank=0,
                    ep_size=1,
                    num_local_experts=2,
                    top_k=6,
                    tokens_per_local_expert=(3, 5),
                    padded_tokens_per_local_expert=(3, 5),
                    gemm_w1=(8, 16, 32),
                    gemm_w3=(8, 16, 32),
                    gemm_w2=(8, 32, 16),
                    dtype="bfloat16",
                    dispatcher="alltoall",
                )
            )
            collector.flush()
            collector.begin_step(2)
            collector.record(
                GroupedGemmRecord(
                    step=2,
                    layer_id=0,
                    micro_batch_id=0,
                    rank=0,
                    ep_rank=0,
                    ep_size=1,
                    num_local_experts=2,
                    top_k=6,
                    tokens_per_local_expert=(3, 7),
                    padded_tokens_per_local_expert=(3, 7),
                    gemm_w1=(10, 16, 32),
                    gemm_w3=(10, 16, 32),
                    gemm_w2=(10, 32, 16),
                    dtype="bfloat16",
                    dispatcher="alltoall",
                )
            )
            collector.flush()
            collector.close()

            hist_file = Path(tmpdir) / "moe_metrics" / "m_histogram_rank_0.csv"
            self.assertTrue(hist_file.exists())
            rows = hist_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(rows[0], "layer_id,M,count")
            # M=3 appears in both steps -> count 2; M=5 and M=7 once each.
            counts = {
                (int(r.split(",")[0]), int(r.split(",")[1])): int(r.split(",")[2])
                for r in rows[1:]
            }
            self.assertEqual(counts[(0, 3)], 2)
            self.assertEqual(counts[(0, 5)], 1)
            self.assertEqual(counts[(0, 7)], 1)

            # Per-rank summary stats over the weighted M distribution.
            summary_file = Path(tmpdir) / "moe_metrics" / "m_summary_rank_0.csv"
            self.assertTrue(summary_file.exists())
            summary_rows = summary_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(summary_rows[0], "layer_id,count,min,max,mean,median")
            summary = {r.split(",")[0]: r.split(",")[1:] for r in summary_rows[1:]}
            # Layer 0: M values {3, 3, 5, 7} -> count 4, min 3, max 7, mean 4.5, median 3.
            self.assertEqual(summary["0"], ["4", "3", "7", "4.5", "3"])
            # Single layer, so the aggregate "all" row matches layer 0.
            self.assertEqual(summary["all"], ["4", "3", "7", "4.5", "3"])

            # No per-record JSONL when only the histogram sink is selected.
            self.assertFalse((Path(tmpdir) / "moe_metrics" / "rank_0.jsonl").exists())

    def test_manifest_records_invariant_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=True, sinks=["histogram"]),
                dump_folder=tmpdir,
                rank=0,
                world_size=8,
                run_metadata={
                    "model_name": "deepseek_v3",
                    "seq_len": 4096,
                    "local_batch_size": 8,
                    "ep": 8,
                    "tp": 8,
                },
            )
            collector.begin_step(1)
            collector.record(_make_record(step=1))
            collector.flush()
            collector.close()

            manifest_file = Path(tmpdir) / "moe_metrics" / "manifest.json"
            self.assertTrue(manifest_file.exists())
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_name"], "deepseek_v3")
            self.assertEqual(manifest["seq_len"], 4096)
            self.assertEqual(manifest["local_batch_size"], 8)
            self.assertEqual(manifest["ep"], 8)
            self.assertEqual(manifest["world_size"], 8)
            # Invariant (N, K) templates captured once from the records.
            templates = manifest["gemm_templates"]
            self.assertEqual(templates["w1"], {"N": 32, "K": 16})
            self.assertEqual(templates["w2"], {"N": 16, "K": 32})
            self.assertEqual(templates["num_local_experts"], 2)
            self.assertEqual(templates["top_k"], 6)
            self.assertEqual(templates["dispatcher"], "alltoall")


class TestImbalanceStats(unittest.TestCase):
    def _read_csv(self, path: Path) -> tuple[list[str], list[list[str]]]:
        rows = path.read_text(encoding="utf-8").strip().splitlines()
        header = rows[0].split(",")
        data = [r.split(",") for r in rows[1:]]
        return header, data

    def test_intra_and_inter_rank_imbalance_single_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MoEMetricCollector(
                config=MoEMetricsConfig(enabled=True, sinks=["histogram"]),
                dump_folder=tmpdir,
                rank=0,
                world_size=1,
            )
            # Step 1, layer 0: experts (3, 5). Step 2, layer 0: experts (3, 7).
            # rank=0 -> expert identity pairs are (rank=0, local). The smaller
            # token count (3) is local 0 -> "0/0", the larger is local 1 ->
            # "0/1".
            for step, tokens in ((1, (3, 5)), (2, (3, 7))):
                collector.begin_step(step)
                collector.record(
                    GroupedGemmRecord(
                        step=step,
                        layer_id=0,
                        micro_batch_id=0,
                        rank=0,
                        ep_rank=1,
                        ep_size=2,
                        num_local_experts=2,
                        top_k=6,
                        tokens_per_local_expert=tokens,
                        padded_tokens_per_local_expert=tokens,
                        gemm_w1=(8, 16, 32),
                        gemm_w3=(8, 16, 32),
                        gemm_w2=(8, 32, 16),
                        dtype="bfloat16",
                        dispatcher="alltoall",
                    )
                )
                collector.flush()
            collector.close()

            moe_dir = Path(tmpdir) / "moe_metrics"
            # Intra-rank per-(step, layer) stats.
            header, data = self._read_csv(moe_dir / "m_imbalance_rank_0.csv")
            self.assertEqual(
                header,
                [
                    "step",
                    "layer_id",
                    "count",
                    "min_expert",
                    "max_expert",
                    "min",
                    "max",
                    "mean",
                    "std",
                    "cv",
                    "load",
                ],
            )
            by_step = {row[0]: row for row in data}
            # Step 1: {3, 5} -> count 2, min_expert 0/0, max_expert 0/1,
            # min 3, max 5, mean 4, std 1, cv 0.25, load 8.
            self.assertEqual(
                by_step["1"][2:],
                ["2", "0/0", "0/1", "3", "5", "4.0", "1.0", "0.25", "8"],
            )
            # Step 2: {3, 7} -> count 2, min_expert 0/0, max_expert 0/1,
            # min 3, max 7, mean 5, std 2, cv 0.4, load 10.
            self.assertEqual(
                by_step["2"][2:],
                ["2", "0/0", "0/1", "3", "7", "5.0", "2.0", "0.4", "10"],
            )

            # Inter-rank file: single rank -> imbalance factor 1.0.
            header_g, data_g = self._read_csv(moe_dir / "m_imbalance_global.csv")
            self.assertEqual(
                header_g,
                [
                    "step",
                    "layer_id",
                    "num_ranks",
                    "min_expert",
                    "min_expert_m",
                    "max_expert",
                    "max_expert_m",
                    "load_min",
                    "load_min_rank",
                    "load_max",
                    "load_max_rank",
                    "load_mean",
                    "load_std",
                    "load_cv",
                    "max_over_mean",
                ],
            )
            global_by_step = {row[0]: row for row in data_g}
            # Step 1: coldest expert 0/0 (m=3), hottest 0/1 (m=5); num_ranks 1,
            # load min=max=mean=8 (both on rank 0), std 0, cv 0, max/mean 1.
            self.assertEqual(
                global_by_step["1"][2:],
                [
                    "1",
                    "0/0",
                    "3",
                    "0/1",
                    "5",
                    "8",
                    "0",
                    "8",
                    "0",
                    "8.0",
                    "0.0",
                    "0.0",
                    "1.0",
                ],
            )
            # Step 2: coldest 0/0 (m=3), hottest 0/1 (m=7); load 10.
            self.assertEqual(
                global_by_step["2"][2:],
                [
                    "1",
                    "0/0",
                    "3",
                    "0/1",
                    "7",
                    "10",
                    "0",
                    "10",
                    "0",
                    "10.0",
                    "0.0",
                    "0.0",
                    "1.0",
                ],
            )


class TestTensorBoardMoESink(unittest.TestCase):
    def _make_sink(self, tmpdir):
        from torchtitan.components.moe_metrics import TensorBoardMoESink

        return TensorBoardMoESink(
            log_dir=str(Path(tmpdir) / "tb"),
            rank=0,
            world_size=1,
            cross_rank_gather=False,
            run_metadata={"model_name": "deepseek_v3", "ep": 1},
        )

    def _record(self, step, rank=0, ep_rank=0, ep_size=1, tokens=(3, 5), layer_id=1):
        return GroupedGemmRecord(
            step=step,
            layer_id=layer_id,
            micro_batch_id=0,
            rank=rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_local_experts=len(tokens),
            top_k=6,
            tokens_per_local_expert=tokens,
            padded_tokens_per_local_expert=tokens,
            gemm_w1=(1, 16, 32),
            gemm_w3=(1, 16, 32),
            gemm_w2=(1, 32, 16),
            dtype="bfloat16",
            dispatcher="alltoall",
        )

    def test_assemble_cube_maps_local_experts_to_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            sink.write_record(self._record(1, tokens=(1, 11)))
            sink.write_record(self._record(2, tokens=(2, 12)))
            payload = (sink._world_rank, sink._num_local_experts, sink._loads)
            cube, steps, layers, n_local = sink._assemble_cube([payload])
            self.assertEqual(steps, [1, 2])
            self.assertEqual(layers, [1])
            self.assertEqual(n_local, 2)
            self.assertEqual(cube.shape, (2, 1, 2))
            # Row 0 (local 0) -> first token; row 1 (local 1) -> second.
            self.assertEqual(list(cube[:, 0, 0]), [1, 11])
            self.assertEqual(list(cube[:, 0, 1]), [2, 12])

    def test_assemble_cube_offsets_by_world_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            # Two ranks' payloads keyed by world rank 0 and 1, 2 local each.
            payloads = [
                (0, 2, {(1, 1, 0): 3, (1, 1, 1): 5}),
                (1, 2, {(1, 1, 0): 7, (1, 1, 1): 9}),
            ]
            cube, steps, layers, n_local = sink._assemble_cube(payloads)
            self.assertEqual(cube.shape, (4, 1, 1))  # 2 ranks x 2 local = 4
            self.assertEqual(list(cube[:, 0, 0]), [3, 5, 7, 9])

    def test_assemble_cube_separates_replica_ep_groups(self):
        # TP=4/EP=4 on 8 ranks => 2 EP groups: ep_rank 0..3 repeats across
        # ranks 0..3 and 4..7. Keying by world rank must keep all 8 ranks on
        # distinct row bands instead of overwriting (the ep_rank-keyed bug).
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            payloads = []
            for world_rank in range(8):
                load = world_rank + 1  # distinct per rank so overwrite is visible
                payloads.append((world_rank, 2, {(1, 1, 0): load, (1, 1, 1): load}))
            cube, _, _, n_local = sink._assemble_cube(payloads)
            self.assertEqual(n_local, 2)
            self.assertEqual(cube.shape, (16, 1, 1))  # 8 ranks x 2 local = 16
            # Every rank's load survives on its own band (no collision).
            expected = [r + 1 for r in range(8) for _ in range(2)]
            self.assertEqual(list(cube[:, 0, 0]), expected)

    def test_assemble_cube_empty_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            cube, _, _, _ = sink._assemble_cube([(None, None, {})])
            self.assertIsNone(cube)

    def test_manifest_markdown_includes_metadata_and_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            sink.set_gemm_templates({"w1": {"N": 32, "K": 16}, "top_k": 6})
            sink.set_dense_gemm_templates(
                {"layers.*.attention.wq": {"N": 64, "K": 64, "count": 4}}
            )
            md = sink._manifest_markdown()
            self.assertIn("### Run metadata", md)
            self.assertIn("| `model_name` | deepseek_v3 |", md)
            self.assertIn("#### gemm_templates", md)
            self.assertIn("top_k", md)
            self.assertIn("#### dense_gemm_templates", md)
            self.assertIn("layers.*.attention.wq", md)

    def test_close_writes_tb_event_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            sink.write_record(self._record(1))
            sink.close()
            events = list((Path(tmpdir) / "tb").glob("events.out.tfevents.*"))
            self.assertTrue(events, "expected a TensorBoard event file")

    def test_close_logs_run_metadata_without_moe_records(self):
        # Dense / non-MoE networks produce no grouped-GEMM records, so the cube
        # is empty. The run-metadata text must still be logged for any network.
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            # No write_record calls: simulate a dense model.
            sink.close()
            ea = EventAccumulator(str(Path(tmpdir) / "tb"))
            ea.Reload()
            text_tags = set(ea.Tags().get("tensors", []))
            self.assertIn("run_metadata/text_summary", text_tags)
            # No cube-derived artifacts when there are no MoE records.
            self.assertEqual(ea.Tags()["images"], [])
            self.assertEqual(ea.Tags()["scalars"], [])

    def test_close_logs_per_layer_spread_scalars(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            # Two experts with loads 3 and 5 at layer 1, step 1.
            sink.write_record(self._record(1, tokens=(3, 5), layer_id=1))
            sink.close()
            ea = EventAccumulator(str(Path(tmpdir) / "tb"))
            ea.Reload()
            scalar_tags = set(ea.Tags()["scalars"])
            for metric in (
                "tokens_max",
                "tokens_min",
                "tokens_range",
                "tokens_mean",
                "max_over_mean",
            ):
                self.assertIn(f"moe_expert_load/layer_1/{metric}", scalar_tags)
            self.assertEqual(
                ea.Scalars("moe_expert_load/layer_1/tokens_max")[0].value, 5
            )
            self.assertEqual(
                ea.Scalars("moe_expert_load/layer_1/tokens_min")[0].value, 3
            )
            self.assertEqual(
                ea.Scalars("moe_expert_load/layer_1/tokens_range")[0].value, 2
            )
            self.assertEqual(
                ea.Scalars("moe_expert_load/layer_1/tokens_mean")[0].value, 4
            )

    def test_close_logs_layer_spread_curve_images(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = self._make_sink(tmpdir)
            sink.write_record(self._record(1, tokens=(3, 5), layer_id=1))
            sink.write_record(self._record(2, tokens=(4, 8), layer_id=1))
            sink.close()
            ea = EventAccumulator(str(Path(tmpdir) / "tb"))
            ea.Reload()
            image_tags = set(ea.Tags()["images"])
            for metric in (
                "tokens_max",
                "tokens_min",
                "tokens_range",
                "tokens_mean",
            ):
                self.assertIn(f"moe_layer_spread/{metric}", image_tags)


class TestModelMetadataExtraction(unittest.TestCase):
    def test_extracts_dim_layers_and_scalar_attention_fields(self):
        from dataclasses import dataclass, field

        from torchtitan.components.metrics import _extract_model_metadata

        @dataclass
        class _Nested:
            in_features: int = 8

        @dataclass
        class _Attn:
            n_heads: int = 16
            kv_lora_rank: int = 512
            mask_type: str = "causal"
            use_rope: bool = True
            rope_factor: float = 1.0
            # Nested sub-config (e.g. a Linear projection) must be skipped.
            wq: _Nested = field(default_factory=_Nested)

        @dataclass
        class _Layer:
            attention: _Attn = field(default_factory=_Attn)

        @dataclass
        class _Model:
            dim: int = 2048
            layers: list = field(default_factory=lambda: [_Layer(), _Layer()])

        meta = _extract_model_metadata(_Model())
        self.assertEqual(meta["dim"], 2048)
        self.assertEqual(meta["num_layers"], 2)
        self.assertEqual(
            meta["attention"],
            {
                "n_heads": 16,
                "kv_lora_rank": 512,
                "mask_type": "causal",
                "use_rope": True,
                "rope_factor": 1.0,
            },
        )
        self.assertNotIn("wq", meta["attention"])

    def test_returns_empty_for_none_or_unstructured_config(self):
        from torchtitan.components.metrics import _extract_model_metadata

        self.assertEqual(_extract_model_metadata(None), {})
        self.assertEqual(_extract_model_metadata(object()), {})


class TestDenseGemmTemplates(unittest.TestCase):
    def _model(self) -> torch.nn.Module:
        # Two "layers" each with a same-shaped attention projection plus a
        # differently-shaped FFN, and a single output head. Mirrors the dense
        # GEMMs of a transformer block.
        def block():
            return torch.nn.ModuleDict(
                {
                    "wq": torch.nn.Linear(64, 64, bias=False),
                    "w1": torch.nn.Linear(64, 128, bias=False),
                }
            )

        return torch.nn.ModuleDict(
            {
                "layers": torch.nn.ModuleList([block(), block()]),
                "output": torch.nn.Linear(64, 100, bias=False),
                "norm": torch.nn.LayerNorm(64),  # not a Linear -> skipped
            }
        )

    def test_dedups_by_normalized_name_with_count(self):
        templates = collect_dense_gemm_templates(
            [self._model()], compute_dtype="bfloat16"
        )
        # Two layers share wq and w1 -> collapse to one template each, count=2.
        self.assertEqual(
            templates["layers.*.wq"],
            {
                "N": 64,
                "K": 64,
                "param_dtype": "float32",
                "compute_dtype": "bfloat16",
                "count": 2,
            },
        )
        self.assertEqual(
            templates["layers.*.w1"],
            {
                "N": 128,
                "K": 64,
                "param_dtype": "float32",
                "compute_dtype": "bfloat16",
                "count": 2,
            },
        )
        # Output head appears once.
        self.assertEqual(
            templates["output"],
            {
                "N": 100,
                "K": 64,
                "param_dtype": "float32",
                "compute_dtype": "bfloat16",
                "count": 1,
            },
        )
        # LayerNorm is not an nn.Linear and must be excluded.
        self.assertNotIn("norm", templates)

    def test_compute_dtype_falls_back_to_param_dtype(self):
        templates = collect_dense_gemm_templates([self._model()])
        # No compute_dtype supplied -> compute_dtype mirrors the stored dtype.
        self.assertEqual(templates["output"]["param_dtype"], "float32")
        self.assertEqual(templates["output"]["compute_dtype"], "float32")

    def test_disambiguates_same_name_different_shape(self):
        m = torch.nn.ModuleDict(
            {
                "layers": torch.nn.ModuleList(
                    [
                        torch.nn.Linear(8, 16, bias=False),
                        torch.nn.Linear(8, 32, bias=False),  # same norm name, diff N
                    ]
                )
            }
        )
        templates = collect_dense_gemm_templates([m])
        self.assertEqual(templates["layers.*"]["N"], 16)
        self.assertEqual(templates["layers.*#2"]["N"], 32)

    def test_empty_for_model_without_linears(self):
        m = torch.nn.ModuleDict({"norm": torch.nn.LayerNorm(8)})
        self.assertEqual(collect_dense_gemm_templates([m]), {})


if __name__ == "__main__":
    unittest.main()
