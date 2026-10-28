# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from torchtitan.observability.metrics import (
    BaseLogger,
    LoggerContainer,
    MetricsProcessor,
    MLFlowLogger,
)


@pytest.fixture
def fake_mlflow(monkeypatch):
    client = MagicMock()
    client.active_run.return_value = None
    client.start_run.return_value.info.run_id = "run-123"
    monkeypatch.setitem(sys.modules, "mlflow", client)
    return client


def test_metrics_run_and_status(fake_mlflow):
    metric_logger = MLFlowLogger(config_dict={"steps": 10}, tag="train")
    fake_mlflow.start_run.assert_called_once_with()

    metric_logger.log({"throughput(tps)": 42, "mfu(%)": 35}, step=2)
    metric_logger.log({"validation/loss": 0.5}, step=3)

    fake_mlflow.start_run.assert_called_once_with()
    fake_mlflow.log_dict.assert_called_once_with(
        {"steps": 10}, "torchtitan_config.json"
    )
    assert fake_mlflow.log_metrics.call_args_list[0].kwargs == {"step": 2}
    assert fake_mlflow.log_metrics.call_args_list[0].args[0] == {
        "train/throughput_tps": 42,
        "train/mfu_percent": 35,
    }
    assert fake_mlflow.log_metrics.call_args_list[1].kwargs == {"step": 3}
    metric_logger.close()
    metric_logger.close()
    fake_mlflow.end_run.assert_called_once_with(status="FINISHED")


def test_metric_name_collision(fake_mlflow):
    metric_logger = MLFlowLogger()
    with pytest.raises(ValueError, match="collide"):
        metric_logger.log({"mfu(%)": 35, "mfu_percent": 36}, step=1)
    metric_logger.close()


def test_failed_run(fake_mlflow):
    metric_logger = MLFlowLogger()
    try:
        raise RuntimeError("training failed")
    except RuntimeError:
        metric_logger.close()
    fake_mlflow.end_run.assert_called_once_with(status="FAILED")


def test_existing_run_is_not_closed(fake_mlflow):
    fake_mlflow.active_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="existing")
    )
    metric_logger = MLFlowLogger()
    metric_logger.log({"loss": 0.5}, step=1)
    metric_logger.close()
    fake_mlflow.start_run.assert_not_called()
    fake_mlflow.set_tags.assert_not_called()
    fake_mlflow.end_run.assert_not_called()


@pytest.mark.parametrize("save_for_all_ranks", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_only_metrics_rank_uses_mlflow(
    fake_mlflow, monkeypatch, save_for_all_ranks, rank
):
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    processor = object.__new__(MetricsProcessor)
    result = processor._build_metric_logger(
        config=MetricsProcessor.Config(
            enable_mlflow=True, save_for_all_ranks=save_for_all_ranks
        ),
        parallelism_context=SimpleNamespace(pp_enabled=False),
        dump_folder="unused",
        pp_schedule="1F1B",
    )
    result.log({"loss": 1.0}, step=1)
    assert fake_mlflow.start_run.call_count == (1 if rank == 0 else 0)
    result.close()


def test_disabled_does_not_import_mlflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    processor = object.__new__(MetricsProcessor)
    result = processor._build_metric_logger(
        config=MetricsProcessor.Config(),
        parallelism_context=SimpleNamespace(pp_enabled=False),
        dump_folder="unused",
        pp_schedule="1F1B",
    )
    assert type(result) is BaseLogger


def test_missing_dependency_fails_before_rank_selection(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    processor = object.__new__(MetricsProcessor)
    with pytest.raises(ImportError, match=r"torchtitan\[mlflow\]"):
        processor._build_metric_logger(
            config=MetricsProcessor.Config(enable_mlflow=True),
            parallelism_context=SimpleNamespace(pp_enabled=False),
            dump_folder="unused",
            pp_schedule="1F1B",
        )


@pytest.mark.parametrize("rank", [0, 4])
def test_pipeline_parallel_uses_loss_rank(fake_mlflow, monkeypatch, rank):
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    monkeypatch.setattr(
        "torchtitan.observability.metrics._is_v_schedule", lambda _: False
    )
    processor = object.__new__(MetricsProcessor)
    result = processor._build_metric_logger(
        config=MetricsProcessor.Config(enable_mlflow=True, save_for_all_ranks=True),
        parallelism_context=SimpleNamespace(pp_enabled=True, world_size=8, pp=2),
        dump_folder="unused",
        pp_schedule="1F1B",
    )
    result.log({"loss": 1.0}, step=1)
    assert fake_mlflow.start_run.call_count == (1 if rank == 4 else 0)
    result.close()


def test_container_closes_failed_run(fake_mlflow):
    container = LoggerContainer()
    container.add_logger(MLFlowLogger())
    container.log({"loss": 1.0}, step=1)
    try:
        raise RuntimeError("training failed")
    except RuntimeError:
        container.close()
    fake_mlflow.end_run.assert_called_once_with(status="FAILED")


def test_local_tracking_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    try:
        metric_logger = MLFlowLogger(config_dict={"training": {"steps": 3}})
        metric_logger.log({"loss": 1.25, "mfu(%)": 42}, step=1)
        run_id = mlflow.active_run().info.run_id
        metric_logger.log({"loss": 0.75, "validation_metrics/nll": 0.6}, step=3)
        metric_logger.close()

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.metrics["mfu_percent"] == 42
        assert run.data.metrics["validation_metrics/nll"] == 0.6
        assert [
            (m.step, m.value) for m in client.get_metric_history(run_id, "loss")
        ] == [
            (1, 1.25),
            (3, 0.75),
        ]
        assert [artifact.path for artifact in client.list_artifacts(run_id)] == [
            "torchtitan_config.json"
        ]
    finally:
        mlflow.set_tracking_uri(old_uri)
