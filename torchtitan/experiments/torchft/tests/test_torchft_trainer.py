# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from functools import partial
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.validator import Validator
from torchtitan.config import override
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.distributed import DistributedTopology, ParallelismContext
from torchtitan.models.common import aux_loss
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.llama3 import build_model_config
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_config,
)
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


def test_decoder_accepts_wrapped_transformer_blocks() -> None:
    class WrappedBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = SimpleNamespace(inner_attention=torch.nn.Identity())
            self.attention_metadata: object | None = None

        def forward(
            self,
            x: torch.Tensor,
            attention_metadata: object,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None,
            aux_loss_denominator: torch.Tensor | None,
        ) -> torch.Tensor:
            del aux_loss_denominator
            self.attention_metadata = attention_metadata
            return x + 1

    model = object.__new__(Decoder)
    torch.nn.Module.__init__(model)
    model.tok_embeddings = torch.nn.Identity()
    wrapped_block = WrappedBlock()
    model.layers = torch.nn.ModuleDict({"0": wrapped_block})
    model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    model._skip_lm_head = False

    tokens = torch.zeros(2, 3)
    output = model(
        tokens,
        positions=torch.arange(2),
        attention_metadata=cast(Any, {torch.nn.Identity: "metadata"}),
    )

    torch.testing.assert_close(output, tokens + 1)
    assert wrapped_block.attention_metadata == "metadata"


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})

    @override(target=DenseFeedForward.Config)
    def ffn_lora(config):
        return LoRATransform(rank=1, alpha=1.0).transform(config)

    config = ft.FaultTolerantTrainer.Config(
        model=build_model_config(
            "debugmodel",
            seq_len=2048,
            attn_backend="flex",
            parallelism=ParallelismConfig(),
        ),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
    )
    config.override.imports = [f"{__name__}.ffn_lora"]

    def initialize_distributed_runtime(engine):
        engine.device = torch.device("cpu")
        engine.parallelism_context = ParallelismContext.from_config(
            config.parallelism,
            DistributedTopology(world_size=1),
            dump_folder=config.dump_folder,
        )
        engine.ft_manager = config.fault_tolerance.build()
        engine.garbage_collector = None
        engine.device_memory_monitor = SimpleNamespace()

    class ModelBuildReachedError(Exception):
        """Stop FT initialization at the model-build boundary."""

    built_ffns = []

    def build_model(model_config):
        built_ffns.append(model_config.layers[0].feed_forward.build())
        raise ModelBuildReachedError

    # Bypass hardware/data setup, but keep config updates and FFN building real.
    monkeypatch.setattr(
        ft.FaultTolerantTrainingEngine,
        "_initialize_distributed_runtime",
        initialize_distributed_runtime,
    )
    monkeypatch.setattr(
        ParallelismContext,
        "activate_spmd",
        lambda self: nullcontext(),
    )
    monkeypatch.setattr(
        type(config.dataloader),
        "build",
        lambda self, **kwargs: SimpleNamespace(max_num_documents=None),
    )
    monkeypatch.setattr(
        type(config.metrics),
        "build",
        lambda self, **kwargs: SimpleNamespace(color=""),
    )
    monkeypatch.setattr(type(config.model), "build", build_model)

    with pytest.raises(ModelBuildReachedError):
        ft.FaultTolerantTrainer(config)

    assert hasattr(built_ffns[0].w13, "lora_a"), "FT ignored the FFN LoRA override"


def test_ft_accepts_validation_with_training_accelerator_graphs():
    parallelism = ParallelismConfig()
    config = ft.FaultTolerantTrainer.Config(
        model=build_muse_glimmer_config(
            "debugmodel", seq_len=2048, attn_backend="flex", parallelism=parallelism
        ),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
        parallelism=parallelism,
    )
    config.validator = Validator.Config(steps=1)
    config.training.disable_accelerator_graphs = False
    config.__post_init__()


def test_ft_accepts_eager_validation():
    parallelism = ParallelismConfig(pipeline_parallel_degree=2)
    config = ft.FaultTolerantTrainer.Config(
        model=build_muse_glimmer_config(
            "debugmodel", seq_len=2048, attn_backend="flex", parallelism=parallelism
        ),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
        parallelism=parallelism,
    )
    config.training.disable_accelerator_graphs = True
    config.validator = Validator.Config(steps=1)
    config.__post_init__()


def test_ft_rejects_accelerator_graphed_fsdp_gradient_accumulation(monkeypatch) -> None:
    config = ft.FaultTolerantTrainer.Config(
        model=build_model_config(
            "debugmodel",
            seq_len=2048,
            attn_backend="flex",
            parallelism=ParallelismConfig(),
        ),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
    )
    config.fault_tolerance.enable = True
    config.training.disable_accelerator_graphs = False
    config.training.num_tokens_per_train_step = (
        2 * config.training.num_tokens_per_microbatch_per_dp_rank
    )
    engine = SimpleNamespace(
        parallelism_context=SimpleNamespace(
            dp_enabled=False,
            pp_enabled=False,
            fsdp_enabled=True,
        ),
        ft_manager=SimpleNamespace(get_dp_info=lambda degree, rank: (degree, rank)),
        device_memory_monitor=SimpleNamespace(),
    )

    monkeypatch.setattr(
        ft, "FaultTolerantTrainingEngine", lambda *args, **kwargs: engine
    )
    monkeypatch.setattr(
        type(config.dataloader),
        "build",
        lambda self, **kwargs: SimpleNamespace(max_num_documents=None),
    )
    monkeypatch.setattr(
        type(config.metrics),
        "build",
        lambda self, **kwargs: SimpleNamespace(color=""),
    )
    monkeypatch.setattr(ft, "accelerator_graphs_supported", lambda: True)

    with pytest.raises(
        ValueError,
        match="does not support accelerator-graphed FSDP gradient accumulation",
    ):
        ft.FaultTolerantTrainer(config)


def test_ft_training_engine_rejects_optimizer_accelerator_graph() -> None:
    config = SimpleNamespace(
        optim=SimpleNamespace(enable_accelerator_graph=True),
    )

    with (
        patch.object(TrainingEngine, "__init__") as init,
        pytest.raises(ValueError, match="not supported with TorchFT"),
    ):
        ft.FaultTolerantTrainingEngine(
            config,
            model_config=MagicMock(),
            precision=MagicMock(),
            max_num_documents=None,
            output_dir="",
            fault_tolerance=MagicMock(),
        )

    init.assert_not_called()


@pytest.mark.parametrize("dp_cp_enabled", [True, False])
def test_ft_logs_loss_over_the_loss_tokens_of_every_replica(monkeypatch, dp_cp_enabled):
    # Two active replicas, despite group_size=4.
    replicate_pg = Mock(size=lambda: 2)
    engine = Mock(
        spec=ft.FaultTolerantTrainingEngine,
        config=Mock(training=Mock(disable_accelerator_graphs=True)),
        device=torch.device("cpu"),
        # Size-1 DP and CP meshes: only the replicas reduce.
        parallelism_context=Mock(
            dp_enabled=False,
            dp_cp_enabled=dp_cp_enabled,
            pp_enabled=False,
            ep_enabled=False,
            get_optional_mesh=lambda name: None,
        ),
        ft_manager=Mock(loss_sync_pg=replicate_pg, group_size=4),
        optim=Mock(lr_schedulers=Mock(schedulers=[Mock(get_last_lr=lambda: [0.1])])),
        num_completed_steps=1,
        ntokens_seen=4,
        local_loss_token_count=4,
        # This replica's 4 loss tokens and another replica's 4.
        global_loss_token_count=torch.tensor(8),
        model_config=SimpleNamespace(mtp_layers=None),
        forward_backward=Mock(
            return_value=ForwardBackwardResult(torch.tensor(8.0), [], 400)
        ),
        optim_step=Mock(return_value=torch.tensor(0.0)),
    )
    engine.step_aux_loss_denominators = partial(
        TrainingEngine.step_aux_loss_denominators, engine
    )
    trainer = Mock(
        spec=ft.FaultTolerantTrainer,
        engine=engine,
        metrics_processor=Mock(should_log=Mock(return_value=True)),
        config=Mock(
            dataloader=Mock(num_mtp_layers=0),
            model=Mock(traverse=lambda _: iter((("aux_loss", Mock(), None, None),))),
        ),
        _local_num_flops_since_last_log=0,
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
    )

    # Both active replicas hold this replica's values.
    def replica_sum(x, mesh=None, extra_pg=None):
        return float(x) * (2 if extra_pg is replicate_pg else 1)

    monkeypatch.setattr(ft.dist_utils, "dist_sum", replica_sum)
    monkeypatch.setattr(
        ft.dist_utils, "dist_max", lambda x, mesh=None, extra_pg=None: float(x)
    )
    monkeypatch.setattr(
        aux_loss,
        "all_reduce",
        lambda x, reduceOp, group: x * 2 if group is replicate_pg else x,
    )
    # One aux-loss layer whose value this step is 3.0 on this replica.
    monkeypatch.setattr(AuxLoss, "_group_counts", {("dp", "load_balance"): 1})
    monkeypatch.setattr(
        AuxLoss, "group_acc", {("dp", "load_balance"): torch.tensor(3.0)}
    )

    microbatch = SimpleNamespace(
        labels=torch.arange(4),
        loss_token_count=torch.tensor(4),
        routing_token_counts=torch.tensor([4]),
        as_input_dict=lambda: {
            "positions": torch.arange(4),
            "padding_mask": torch.zeros(4, dtype=torch.bool),
        },
    )
    ft.FaultTolerantTrainer.train_step(trainer, iter([microbatch]))

    engine.forward_backward.assert_called_once()
    forward_backward_args = engine.forward_backward.call_args.kwargs
    assert forward_backward_args.keys() == {
        "microbatch_groups",
        "aux_loss_denominators",
    }
    assert forward_backward_args["microbatch_groups"] == [[microbatch]]
    # 4 routed tokens per 4 loss tokens on this replica.
    torch.testing.assert_close(
        forward_backward_args["aux_loss_denominators"], torch.tensor([1.0])
    )
    trainer.metrics_processor.log.assert_called_once()
    log_call = trainer.metrics_processor.log.call_args
    _, logged_loss, logged_max_loss, _ = log_call.args
    # Both replicas' loss sums over both replicas' loss tokens: (8 + 8) / 8.
    assert logged_loss == 2.0
    # The max over ranks of each rank's sum over its own loss tokens: 8 / 4.
    assert logged_max_loss == 2.0
    # The aux value sums over the same replicas as the count: (3 + 3) / 8.
    assert log_call.kwargs["extra_metrics"]["load_balance/mean"] == 0.75
    assert log_call.kwargs["extra_metrics"]["n_tokens_seen"] == 8


@pytest.mark.parametrize(
    ("dp_degree", "cp_degree", "use_ft_pg"),
    [(1, 1, True), (1, 2, True), (2, 2, True), (1, 2, False), (2, 2, False)],
)
def test_ft_averages_logged_flops_by_active_replica_count(
    monkeypatch, dp_degree, cp_degree, use_ft_pg
):
    dp_cp_enabled = dp_degree * cp_degree > 1
    loss_mesh = SimpleNamespace(size=lambda: dp_degree * cp_degree)
    dp_mesh = SimpleNamespace(size=lambda: dp_degree) if dp_degree > 1 else None
    ft_pg = Mock(size=lambda: 2) if use_ft_pg else None
    meshes = {"dp": dp_mesh, "loss": loss_mesh}
    engine = Mock(
        spec=ft.FaultTolerantTrainingEngine,
        config=Mock(training=Mock(disable_accelerator_graphs=True)),
        device=torch.device("cpu"),
        parallelism_context=Mock(
            dp_enabled=dp_degree > 1,
            dp_cp_enabled=dp_cp_enabled,
            pp_enabled=False,
            ep_enabled=False,
            dp_replicate=1,
            dp_shard=dp_degree,
            cp=cp_degree,
            get_mesh=Mock(side_effect=meshes.__getitem__),
            get_optional_mesh=Mock(side_effect=meshes.__getitem__),
        ),
        ft_manager=Mock(loss_sync_pg=ft_pg, group_size=4),
        optim=Mock(lr_schedulers=Mock(schedulers=[Mock(get_last_lr=lambda: [0.1])])),
        num_completed_steps=1,
        ntokens_seen=4,
        local_loss_token_count=4,
        global_loss_token_count=torch.tensor(4),
        model_config=SimpleNamespace(mtp_layers=None),
        forward_backward=Mock(
            return_value=ForwardBackwardResult(torch.tensor(2.0), [], 400)
        ),
        optim_step=Mock(return_value=torch.tensor(0.0)),
    )
    trainer = Mock(
        spec=ft.FaultTolerantTrainer,
        engine=engine,
        metrics_processor=Mock(should_log=Mock(return_value=True)),
        config=Mock(
            dataloader=Mock(num_mtp_layers=0),
            model=Mock(traverse=lambda _: iter((("aux_loss", Mock(), None, None),))),
        ),
        _local_num_flops_since_last_log=0,
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
    )
    # When enabled, the FT group has two active replicas, despite group_size=4.
    monkeypatch.setattr(
        ft.dist_utils,
        "dist_sum",
        Mock(side_effect=[4.0, 8, 400 * dp_degree * (2 if use_ft_pg else 1)]),
    )
    monkeypatch.setattr(ft.dist_utils, "dist_max", Mock(return_value=2.0))
    monkeypatch.setattr(ft, "collect_aux_loss_metrics", Mock(return_value={}))

    ft.FaultTolerantTrainer.train_step(trainer, iter([SimpleNamespace()]))

    assert trainer.metrics_processor.log.call_args.kwargs["num_flops"] == 400
    reduces_flops = dp_degree > 1 or use_ft_pg
    # The loss sum and the seen tokens, then the FLOPs over the DP mesh and the FT group.
    assert ft.dist_utils.dist_sum.call_count == 2 + reduces_flops
    if reduces_flops:
        _, reduced_mesh, reduced_ft_pg = ft.dist_utils.dist_sum.call_args.args
        assert reduced_mesh is dp_mesh
        assert reduced_ft_pg is ft_pg


@pytest.mark.parametrize("dp_enabled", [False, True])
def test_ft_engine_counts_loss_tokens_over_the_replicas(monkeypatch, dp_enabled):
    engine = object.__new__(ft.FaultTolerantTrainingEngine)
    dp_mesh, replicate_pg = object(), object()
    engine.parallelism_context = SimpleNamespace(
        dp_enabled=dp_enabled, get_mesh=lambda name: {"dp": dp_mesh}[name]
    )
    engine.ft_manager = SimpleNamespace(loss_sync_pg=replicate_pg)

    # 2 DP ranks and 3 replicas, each with this rank's count.
    def fake_dist_sum_tensor(count, mesh=None, extra_pg=None):
        dp_ranks = 2 if mesh is dp_mesh else 1
        replicas = 3 if extra_pg is replicate_pg else 1
        return count * dp_ranks * replicas

    monkeypatch.setattr(ft.dist_utils, "dist_sum_tensor", fake_dist_sum_tensor)

    count = engine._sum_loss_token_count(torch.tensor(4))

    # The all-reduce hook sums gradients over the replicas, so the count covers them too.
    assert int(count) == (4 * 2 * 3 if dp_enabled else 4 * 3)


def test_ft_engine_installs_all_reduce_hook_after_model_initialization() -> None:
    engine = object.__new__(ft.FaultTolerantTrainingEngine)
    engine.model_parts = [object()]
    engine.ft_manager = SimpleNamespace(maybe_set_all_reduce_hook=MagicMock())

    with patch.object(TrainingEngine, "_initialize_model") as initialize_model:
        ft.FaultTolerantTrainingEngine._initialize_model(
            engine,
            hf_assets_path="",
        )

    initialize_model.assert_called_once()
    engine.ft_manager.maybe_set_all_reduce_hook.assert_called_once_with(
        engine.model_parts
    )
