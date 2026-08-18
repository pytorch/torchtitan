# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from functorch.compile import make_boxed_func
from torch import nn
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch.distributed.tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.components.metrics import MetricsProcessor

from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    is_cudagraph_compatible,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import tag_sac_policy
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3.model import Llama3TransformerBlock
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.observability.tensor_logging import (
    init as tensor_logging_init,
    log_fwd_bwd_stats,
    log_stats,
    register,
    register_fwd_bwd,
    set_enabled,
)
from torchtitan.observability.tensor_logging.runtime import (
    _wrap_fwd_bwd_for_tensor_logging_capture,
    should_run_logging_calls,
)
from torchtitan.observability.tensor_logging.statistics import (
    accumulate_tensor_statistics,
    StatisticBuffers,
)


def init(model_parts, *, device=torch.device("cpu"), **kwargs):
    return tensor_logging_init(model_parts, device=device, **kwargs)


@pytest.fixture
def cpu_device_mesh(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


class TinyStatsModule(nn.Module):
    def __init__(self, width: int, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        self.track_forward_calls = track_forward_calls
        self.forward_calls = 0
        register(self, ["hidden"])
        register_fwd_bwd(self, ["output"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.track_forward_calls:
            self.forward_calls += 1
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        output = hidden.square()
        log_fwd_bwd_stats(self, output=output)
        return output


class CompileStatsModule(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        register(self, ["hidden"])
        register_fwd_bwd(self, ["output"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        output = hidden.square()
        log_fwd_bwd_stats(self, output=output)
        return output


class CompileForwardStatsModule(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        register(self, ["hidden"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        return hidden.square()


class TinyStatsRoot(nn.Module):
    def __init__(self, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "0": TinyStatsModule(
                    width=4,
                    track_forward_calls=track_forward_calls,
                )
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](value)


class TinyInputStatsBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))
        register(self, ["hidden"])
        register_fwd_bwd(self, ["input", "output"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        log_fwd_bwd_stats(self, input=value)
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        output = hidden.square()
        log_fwd_bwd_stats(self, output=output)
        return output


class TinyInputStatsRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": TinyInputStatsBlock()})

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](value)


def test_registration_survives_deepcopy() -> None:
    original = TinyStatsRoot()
    copied = copy.deepcopy(original)
    checkpoint_keys = set(copied.state_dict())
    runtime = init(copied)
    try:
        assert set(copied.state_dict()) == checkpoint_keys
        value = torch.randn(3, 4, requires_grad=True)
        with set_enabled(True):
            copied(value).sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["layers.0.hidden"]["counts"][3].item() == 1
        assert snapshot["layers.0.output.x"]["counts"][3].item() == 1
        assert snapshot["layers.0.output.dx"]["counts"][3].item() == 1
    finally:
        runtime.close()


def test_split_roots_share_logical_root_names_and_slots() -> None:
    class PipelinePart(nn.Module):
        def __init__(self, layer_id: int) -> None:
            super().__init__()
            register(self, ["input"])
            self.layers = nn.ModuleDict({str(layer_id): nn.Identity()})
            register(self.layers[str(layer_id)], ["hidden"])

    parts = [PipelinePart(3), PipelinePart(7)]
    runtime = init(parts, pp_enabled=True)
    try:
        assert runtime.full_metric_names == [
            "input",
            "layers.3.hidden",
            "layers.7.hidden",
        ]
        with set_enabled(True):
            log_stats(parts[0], input=torch.ones(2))
        assert runtime.snapshot_unreduced_statistics()["input"]["counts"].tolist() == [
            2,
            0,
            0,
            1,
        ]
    finally:
        runtime.close()


def test_capture_wrapper_preserves_device_cadence(monkeypatch) -> None:
    owner = nn.Module()
    register(owner, ["hidden"])
    runtime = init(owner)
    accumulated_names = []
    accumulate = runtime._accumulate_tensor_statistics

    def track_accumulate(slot, value):
        accumulated_names.append("hidden")
        accumulate(slot, value)

    monkeypatch.setattr(runtime, "_accumulate_tensor_statistics", track_accumulate)
    try:

        def record_hidden() -> torch.Tensor:
            log_stats(owner, hidden=torch.ones(4))
            return torch.ones(())

        wrapped = _wrap_fwd_bwd_for_tensor_logging_capture(record_hidden)
        with set_enabled(False):
            assert not should_run_logging_calls()
            wrapped()

        assert accumulated_names == ["hidden"]
        assert runtime.snapshot_unreduced_statistics()["hidden"]["counts"].tolist() == [
            0,
            0,
            0,
            0,
        ]
    finally:
        runtime.close()


def test_nested_enable_scope_restores_python_and_device_flags() -> None:
    owner = nn.Module()
    runtime = init(owner)
    try:
        with set_enabled(False):
            assert not should_run_logging_calls()
            assert runtime.statistic_buffers.enabled.item() == 0
            with set_enabled(True):
                assert should_run_logging_calls()
                assert runtime.statistic_buffers.enabled.item() == 1
            assert not should_run_logging_calls()
            assert runtime.statistic_buffers.enabled.item() == 0
    finally:
        runtime.close()


def test_init_requires_the_buffer_device() -> None:
    with pytest.raises(TypeError, match="device"):
        cast(Any, tensor_logging_init)(nn.Module())


def test_graph_trainer_rejects_pipeline_tensor_logging_before_setup() -> None:
    config = SimpleNamespace(
        metrics=SimpleNamespace(tensor_logging=SimpleNamespace(enabled=True)),
        parallelism=SimpleNamespace(pipeline_parallel_degree=2),
        compile=SimpleNamespace(precompile_artifact_dir=""),
    )

    with pytest.raises(NotImplementedError, match="GraphPP forward statistics"):
        GraphTrainer(config)


def test_same_owner_duplicate_registration_raises() -> None:
    root = nn.Module()
    register(root, ["value", "value"])
    with pytest.raises(ValueError, match="registered twice: value"):
        init([root], pp_enabled=True)


def test_registration_after_init_is_rejected() -> None:
    root = nn.Module()
    runtime = init(root)
    try:
        with pytest.raises(RuntimeError, match="register tensor names before"):
            register(root, ["late"])
    finally:
        runtime.close()


def test_decoder_input_is_emitted_only_by_the_first_pipeline_stage() -> None:
    for has_embeddings, expected_observations in ((True, 1), (False, 0)):
        decoder = Decoder.__new__(Decoder)
        nn.Module.__init__(decoder)
        decoder.tok_embeddings = nn.Identity() if has_embeddings else None
        decoder.layers = nn.ModuleDict()
        decoder.norm = None
        decoder.lm_head = None
        register_fwd_bwd(decoder, ["input"])

        runtime = init(decoder)
        try:
            value = torch.ones(1, 2, requires_grad=True)
            with set_enabled(True):
                decoder(value).sum().backward()
            assert (
                runtime.snapshot_unreduced_statistics()["input.x"]["counts"][3].item()
                == expected_observations
            )
        finally:
            runtime.close()


class SquareAttention(nn.Module):
    def forward(self, value, attention_masks, positions):
        return value.square()


class ZeroFeedForward(nn.Module):
    def forward(self, value):
        return value * 0


def _residual_block(block_type):
    block = block_type.__new__(block_type)
    nn.Module.__init__(block)
    block.attention = SquareAttention()
    block.attention_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.feed_forward = ZeroFeedForward()
    if block_type is Qwen3TransformerBlock:
        block.moe_enabled = False
    register_fwd_bwd(
        block,
        ["attn_stream", "attn_out", "ffn_stream", "ffn_out"],
    )
    return block


def _trace_forward_backward_step(
    module: nn.Module,
    value: torch.Tensor,
):
    def forward_backward_step(input_value: torch.Tensor) -> list[torch.Tensor]:
        output = module(input_value)
        loss = output.sum()
        gradients = torch.autograd.grad(loss, list(module.parameters()))
        return [loss, *gradients]

    with set_enabled(True):
        return minimal_fx_tracer(
            forward_backward_step,
            module=module,
        )(value)


def _rematerialize_every_forward_node(traced) -> None:
    tag_sac_policy(
        traced.gm,
        policy_fn=lambda node: CheckpointPolicy.MUST_RECOMPUTE,
    )
    selective_activation_remat_pass(traced.gm)


def _assert_snapshots_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key]["counts"], expected[key]["counts"])
        torch.testing.assert_close(actual[key]["sums"], expected[key]["sums"])
        torch.testing.assert_close(actual[key]["maximum"], expected[key]["maximum"])


def _graph_argument_names(graph_module: torch.fx.GraphModule) -> set[str]:
    # AOTAutograd may place effectful custom ops inside functionalization nodes.
    return {
        str(argument)
        for node in graph_module.graph.nodes
        for argument in (node.target, *node.args)
    }


def _run_torchtitan_ac(policy) -> tuple[dict, torch.Tensor, int]:
    torch.manual_seed(0)
    root = TinyStatsRoot()
    block = root.layers["0"]
    if policy is not None:
        policy.build().apply(root)

    value = torch.randn(3, 4, requires_grad=True)
    runtime = init(root)
    try:
        with set_enabled(True):
            root(value).sum().backward()
        return (
            runtime.snapshot_unreduced_statistics(),
            value.grad.clone(),
            block.forward_calls,
        )
    finally:
        runtime.close()


def _run_two_live_graphs(policy) -> tuple[dict, tuple[torch.Tensor, ...], int]:
    torch.manual_seed(0)
    root = TinyStatsRoot()
    block = root.layers["0"]
    if policy is not None:
        policy.build().apply(root)

    values = tuple(torch.randn(3, 4, requires_grad=True) for _ in range(2))
    runtime = init(root)
    try:
        with set_enabled(True):
            outputs = tuple(root(value) for value in values)
            sum(output.sum() for output in outputs).backward()
        gradients = tuple(value.grad.clone() for value in values)
        return runtime.snapshot_unreduced_statistics(), gradients, block.forward_calls
    finally:
        runtime.close()


def test_source_feed_forward_records_act_out_and_cotangent() -> None:
    module = FeedForward.Config(
        w1=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
        w3=Linear.Config(in_features=4, out_features=8),
    ).build()
    runtime = init(module)
    try:
        value = torch.randn(2, 3, 4, requires_grad=True)
        with set_enabled(True):
            module(value).sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["act_out.x"]["counts"].tolist() == [48, 0, 0, 1]
        assert snapshot["act_out.dx"]["counts"].tolist() == [48, 0, 0, 1]
    finally:
        runtime.close()


def test_torchtitan_full_and_selective_ac_record_exactly_once() -> None:
    eager, eager_gradient, eager_calls = _run_torchtitan_ac(None)
    full, full_gradient, full_calls = _run_torchtitan_ac(FullAC.Config())
    selective, selective_gradient, selective_calls = _run_torchtitan_ac(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )

    assert eager_calls == 1
    assert eager["layers.0.hidden"]["counts"].tolist() == [12, 0, 0, 1]
    assert eager["layers.0.output.x"]["counts"].tolist() == [12, 0, 0, 1]
    assert eager["layers.0.output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    assert full_calls == 2
    assert selective_calls == 2
    _assert_snapshots_equal(full, eager)
    _assert_snapshots_equal(selective, eager)
    torch.testing.assert_close(full_gradient, eager_gradient)
    torch.testing.assert_close(selective_gradient, eager_gradient)


def test_checkpointed_block_input_cotangent_is_recorded_once() -> None:
    def run(policy):
        torch.manual_seed(0)
        root = TinyInputStatsRoot()
        if policy is not None:
            policy.build().apply(root)
        value = torch.randn(3, 4, requires_grad=True)
        runtime = init(root)
        try:
            with set_enabled(True):
                root(value).sum().backward()
            return runtime.snapshot_unreduced_statistics(), value.grad.clone()
        finally:
            runtime.close()

    eager, eager_gradient = run(None)
    full, full_gradient = run(FullAC.Config())
    selective, selective_gradient = run(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )

    assert eager["layers.0.input.x"]["counts"].tolist() == [12, 0, 0, 1]
    assert eager["layers.0.input.dx"]["counts"].tolist() == [12, 0, 0, 1]
    _assert_snapshots_equal(full, eager)
    _assert_snapshots_equal(selective, eager)
    torch.testing.assert_close(full_gradient, eager_gradient)
    torch.testing.assert_close(selective_gradient, eager_gradient)


def test_repeated_checkpointed_module_with_two_live_graphs_is_exact() -> None:
    eager, eager_gradients, eager_calls = _run_two_live_graphs(None)
    full, full_gradients, full_calls = _run_two_live_graphs(FullAC.Config())
    selective, selective_gradients, selective_calls = _run_two_live_graphs(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )

    assert eager_calls == 2
    assert full_calls == 4
    assert selective_calls == 4
    assert eager["layers.0.hidden"]["counts"].tolist() == [24, 0, 0, 2]
    assert eager["layers.0.output.x"]["counts"].tolist() == [24, 0, 0, 2]
    assert eager["layers.0.output.dx"]["counts"].tolist() == [24, 0, 0, 2]
    _assert_snapshots_equal(full, eager)
    _assert_snapshots_equal(selective, eager)
    for actual, expected in zip(full_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(selective_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "block_type",
    [Llama3TransformerBlock, Qwen3TransformerBlock],
)
def test_residual_stream_cotangent_includes_skip_and_branch(block_type) -> None:
    block = _residual_block(block_type)
    runtime = init(block)
    try:
        value = torch.tensor([[2.0]], requires_grad=True)
        with set_enabled(True):
            block(value, attention_masks=None).sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["attn_stream.dx"]["sums"][0].item() == 5.0
        assert snapshot["ffn_stream.dx"]["sums"][0].item() == 1.0
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_finite_statistics_match_on_cpu_and_cuda(device: str) -> None:
    owner = nn.Module().to(device)
    register(owner, ["value", "empty"])
    runtime = init(owner, device=torch.device(device))
    try:
        value = torch.tensor(
            [1.0, -2.0, 0.0, torch.nan, torch.inf, -torch.inf],
            device=device,
        )
        with set_enabled(True):
            log_stats(owner, value=value)
            log_stats(owner, empty=torch.empty(0, device=device))

        snapshots = runtime.snapshot_unreduced_statistics()
        assert snapshots["value"]["counts"].tolist() == [6, 3, 1, 1]
        assert snapshots["value"]["sums"].tolist() == [3.0, 5.0, 17.0]
        assert snapshots["value"]["maximum"].item() == 2.0
        assert snapshots["empty"]["counts"].tolist() == [0, 0, 0, 1]
        assert snapshots["empty"]["sums"].tolist() == [0.0, 0.0, 0.0]
        assert snapshots["empty"]["maximum"].item() == -torch.inf
    finally:
        runtime.close()


def test_statistic_buffers_group_fields_by_reduction_operation() -> None:
    buffers = StatisticBuffers(3, device=torch.device("cpu"))

    assert buffers.sum_statistics.shape == (3, 7)
    assert buffers.sum_statistics.dtype == torch.float32
    assert buffers.maxima.shape == (3,)
    assert buffers.maxima.dtype == torch.float32
    assert not hasattr(buffers, "counts")
    assert not hasattr(buffers, "sums")


def test_reducer_issues_one_sum_then_one_max_collective(monkeypatch) -> None:
    owner = nn.Module()
    register(owner, ["value"])
    runtime = init(owner)
    calls = []

    def record_all_reduce(tensor, *, op):
        calls.append((op, tensor.dtype, tuple(tensor.shape)))

    try:
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "all_reduce", record_all_reduce)
        runtime._reduce_buffers()
    finally:
        runtime.close()

    assert calls == [
        (dist.ReduceOp.SUM, torch.float32, (1, 7)),
        (dist.ReduceOp.MAX, torch.float32, (1,)),
    ]


def test_sum_slab_keeps_all_hand_computed_statistics() -> None:
    owner = nn.Module()
    register(owner, ["value"])
    runtime = init(owner)
    try:
        with set_enabled(True):
            log_stats(owner, value=torch.tensor([0.0, 1.0, -2.0, torch.nan]))

        snapshot = runtime.snapshot_unreduced_statistics()["value"]
        assert snapshot["counts"].tolist() == [4, 1, 1, 1]
        assert snapshot["sums"].tolist() == [3.0, 5.0, 17.0]
        assert snapshot["maximum"].item() == 2.0
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("layout", ["transpose", "permute", "strided"])
def test_noncontiguous_cuda_statistics_match_values(layout: str) -> None:
    owner = nn.Module().cuda()
    register(owner, ["value"])
    runtime = init(owner, device=torch.device("cuda"))
    try:
        value = torch.tensor(
            [1.0, -2.0, 0.0, torch.nan, torch.inf, -torch.inf, 4, -5, 6, 0, 7, 8],
            device="cuda",
        ).reshape(2, 3, 2)
        if layout == "transpose":
            value = value.transpose(0, 1)
        elif layout == "permute":
            value = value.permute(2, 0, 1)
        else:
            value = torch.stack((value, torch.zeros_like(value)), dim=-1)[..., 0]
        assert not value.is_contiguous()
        with set_enabled(True):
            log_stats(owner, value=value)

        snapshot = runtime.snapshot_unreduced_statistics()["value"]
        assert snapshot["counts"].tolist() == [12, 3, 2, 1]
        assert snapshot["sums"].tolist() == [33.0, 195.0, 8691.0]
        assert snapshot["maximum"].item() == 8.0
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_statistics_support_64_bit_indexing(monkeypatch) -> None:
    from torchtitan.observability.tensor_logging import statistics_triton

    # Exercise the large-tensor loop and its 64-bit indexing without a 4 GiB fixture.
    monkeypatch.setattr(statistics_triton, "_MAX_PROGRAMS", 1)
    monkeypatch.setattr(statistics_triton, "_MAX_INT32_INDEXED_ELEMENTS", -1)
    value = torch.zeros(4097, dtype=torch.bfloat16, device="cuda")
    value[0] = 2
    value[-1] = 3
    sum_statistics = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
    maximum = torch.full((1,), -torch.inf, dtype=torch.float32, device="cuda")
    enabled = torch.ones((), dtype=torch.int32, device="cuda")
    slot_index = torch.tensor(0)

    accumulate_tensor_statistics(
        value,
        sum_statistics,
        maximum,
        enabled,
        slot_index,
    )

    assert sum_statistics[0].tolist() == [4097, 0, 4095, 1, 5.0, 13.0, 97.0]
    assert maximum[0].item() == 3.0


def test_metrics_filter_matches_name_and_statistic() -> None:
    owner = nn.Module()
    register(owner, ["hidden", "other"])
    runtime = init(
        owner,
        publish_filter_regex=r"^hidden\.(?:abs_mean|abs_max)$",
    )
    try:
        with set_enabled(True):
            log_stats(
                owner,
                hidden=torch.tensor([1.0, -3.0]),
                other=torch.tensor([5.0]),
            )

        metrics = runtime.collect()
        assert metrics == {
            "hidden.abs_mean": 2.0,
            "hidden.abs_max": 3.0,
        }
    finally:
        runtime.close()


def test_default_filter_publishes_nonfinite_count() -> None:
    owner = nn.Module()
    register(owner, ["hidden"])
    runtime = init(
        owner,
        publish_filter_regex=(
            MetricsProcessor.Config().tensor_logging.publish_filter_regex
        ),
    )
    try:
        with set_enabled(True):
            log_stats(owner, hidden=torch.tensor([1.0, torch.nan]))

        metrics = runtime.collect()
        assert metrics["hidden.numel"] == 2
        assert metrics["hidden.nonfinite_count"] == 1
        assert metrics["hidden.abs_mean"] == 1.0
        assert metrics["hidden.square_mean"] == 1.0
        assert metrics["hidden.abs_max"] == 1.0
        assert "hidden.kurtosis" not in metrics
    finally:
        runtime.close()


def test_overflowed_moments_do_not_hide_valid_lower_order_statistics() -> None:
    owner = nn.Module()
    register(owner, ["square_overflow", "fourth_overflow"])
    runtime = init(owner)
    try:
        with set_enabled(True):
            log_stats(
                owner,
                square_overflow=torch.tensor([1e20]),
                fourth_overflow=torch.tensor([1e10]),
            )

        metrics = runtime.collect()
        assert metrics["square_overflow.abs_mean"] == pytest.approx(1e20)
        assert metrics["square_overflow.abs_max"] == pytest.approx(1e20)
        assert "square_overflow.square_mean" not in metrics
        assert "square_overflow.rms" not in metrics
        assert "square_overflow.kurtosis" not in metrics
        assert metrics["fourth_overflow.square_mean"] == pytest.approx(1e20)
        assert metrics["fourth_overflow.rms"] == pytest.approx(1e10)
        assert "fourth_overflow.kurtosis" not in metrics
    finally:
        runtime.close()


def test_registered_but_unobserved_key_is_not_published() -> None:
    owner = nn.Module()
    register(owner, ["observed", "missing"])
    runtime = init(owner)
    try:
        with set_enabled(True):
            log_stats(owner, observed=torch.tensor([2.0]))

        metrics = runtime.collect()
        assert metrics["observed.abs_mean"] == 2.0
        assert not any(key.startswith("missing.") for key in metrics)
    finally:
        runtime.close()


def test_unregistered_emission_names_the_missing_key() -> None:
    owner = nn.Module()
    register(owner, ["known"])
    runtime = init(owner)
    try:
        with (
            set_enabled(True),
            pytest.raises(
                KeyError,
                match="unregistered tensor metric: missing",
            ),
        ):
            log_stats(owner, missing=torch.ones(1))
    finally:
        runtime.close()


def test_generic_counts_follow_float32_integer_precision() -> None:
    owner = nn.Module()
    register(owner, ["value"])
    runtime = init(owner)
    try:
        runtime.statistic_buffers.sum_statistics[0, :4].copy_(
            torch.tensor([2**24 + 1, 0, 0, 1], dtype=torch.float32)
        )
        metrics = runtime.collect()
        assert metrics["value.numel"] == 2**24
        assert isinstance(metrics["value.numel"], int)
    finally:
        runtime.close()


def test_frozen_tensor_records_forward_without_backward_hook() -> None:
    owner = nn.Module()
    register_fwd_bwd(owner, ["value"])
    runtime = init(owner)
    try:
        with set_enabled(True):
            log_fwd_bwd_stats(owner, value=torch.ones(2, 3))

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["value.x"]["counts"].tolist() == [6, 0, 0, 1]
        assert snapshot["value.dx"]["counts"].tolist() == [0, 0, 0, 0]
    finally:
        runtime.close()


@pytest.mark.parametrize("placement", [Replicate(), Shard(0)])
def test_dtensor_records_local_forward_and_cotangent(
    cpu_device_mesh,
    placement,
) -> None:
    owner = nn.Module()
    register_fwd_bwd(owner, ["value"])
    runtime = init(owner)
    try:
        local_value = torch.tensor(
            [[1.0, -2.0], [0.0, 4.0]],
            requires_grad=True,
        )
        value = DTensor.from_local(
            local_value,
            cpu_device_mesh,
            (placement,),
            run_check=False,
        )
        with set_enabled(True):
            log_fwd_bwd_stats(owner, value=value)
            (value * 2).to_local().sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["value.x"]["counts"].tolist() == [4, 0, 1, 1]
        assert snapshot["value.dx"]["counts"].tolist() == [4, 0, 0, 1]
        assert snapshot["value.x"]["sums"].tolist() == [7.0, 21.0, 273.0]
        assert snapshot["value.dx"]["sums"].tolist() == [8.0, 16.0, 64.0]
        assert snapshot["value.x"]["maximum"].item() == 4.0
        assert snapshot["value.dx"]["maximum"].item() == 2.0
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_graph_trainer_trace_remat_replay_and_cadence_are_exact(
    device: str,
) -> None:
    torch.manual_seed(0)
    module = TinyStatsModule(width=4, track_forward_calls=False).to(device)
    value = torch.randn(3, 4, device=device)
    runtime = init(module, device=torch.device(device))
    buffer_addresses = tuple(
        buffer.data_ptr() for buffer in runtime.statistic_buffers.buffers()
    )

    try:
        with set_enabled(True):
            eager_output = module(value)
            eager_loss = eager_output.sum()
            eager_gradients = torch.autograd.grad(
                eager_loss,
                tuple(module.parameters()),
            )
        eager_snapshot = runtime.snapshot_unreduced_statistics()
        runtime.collect()

        traced = _trace_forward_backward_step(module, value)

        assert all(
            count == 0
            for statistic in runtime.snapshot_unreduced_statistics().values()
            for count in statistic["counts"].tolist()
        )

        _rematerialize_every_forward_node(traced)

        runner = run_traced(traced, module=module, _validate_runtime=True)
        with set_enabled(True):
            graph_result = runner(value)
        enabled_snapshot = runtime.snapshot_unreduced_statistics()
        _assert_snapshots_equal(enabled_snapshot, eager_snapshot)
        torch.testing.assert_close(graph_result[0], eager_loss)
        for actual, expected in zip(
            graph_result[1:],
            eager_gradients,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        assert enabled_snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert enabled_snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert enabled_snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]

        runtime.collect()
        with set_enabled(False):
            runner(value)
        disabled_snapshot = runtime.snapshot_unreduced_statistics()
        assert all(
            count == 0
            for statistic in disabled_snapshot.values()
            for count in statistic["counts"].tolist()
        )

        with set_enabled(True):
            runner(value)
            runner(value)
        replay_snapshot = runtime.snapshot_unreduced_statistics()
        assert replay_snapshot["hidden"]["counts"].tolist() == [24, 0, 0, 2]
        assert replay_snapshot["output.x"]["counts"].tolist() == [24, 0, 0, 2]
        assert replay_snapshot["output.dx"]["counts"].tolist() == [24, 0, 0, 2]
        assert tuple(
            buffer.data_ptr() for buffer in runtime.statistic_buffers.buffers()
        ) == (buffer_addresses)
        assert all("_tensor_logging_state" not in key for key in module.state_dict())
    finally:
        runtime.close()


def test_graph_trainer_precompiled_artifact_uses_live_logging_buffers(tmp_path) -> None:
    from torchtitan.experiments.graph_trainer.precompile import (
        precompile_fx_trace_load,
        precompile_fx_trace_save,
    )
    from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

    module = TinyStatsModule(width=4, track_forward_calls=False)
    value = torch.randn(3, 4)
    runtime = init(module)
    try:
        with set_enabled(True):
            traced = _trace_forward_backward_step(module, value)

        storage = DiskStorageAdapter(str(tmp_path))
        precompile_fx_trace_save(traced, storage)
        loaded = precompile_fx_trace_load(storage, expected_fingerprint="")

        assert any(
            name.startswith("_tensor_logging_state.") for name in loaded.state_fqns
        )
        runtime.collect()
        with set_enabled(True):
            run_traced(loaded, module=module)(value)

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_trainer_production_cache_traces_once_across_cadence() -> None:
    module = TinyStatsModule(width=4, track_forward_calls=False).cuda()
    value = torch.randn(3, 4, device="cuda")
    labels = torch.randn_like(value)
    runtime = init(module, device=torch.device("cuda"))
    trainer = object.__new__(GraphTrainer)
    trainer._traced_step = None
    trainer.tensor_logging = runtime
    trainer.loss_fn = (
        lambda prediction, target, **_: (prediction - target).square().sum()
    )
    trainer.train_context = get_spmd_context()
    trainer.config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable=True,
            mode="aot_fx_trace",
            enable_passes=False,
        )
    )
    params = list(module.parameters())
    try:
        with mock.patch(
            "torchtitan.experiments.graph_trainer.trainer.minimal_fx_tracer",
            wraps=minimal_fx_tracer,
        ) as trace_call:
            for step in range(1, 7):
                for parameter in params:
                    parameter.grad = None
                with set_enabled(step % 2 == 0):
                    trainer._make_fx_forward_backward_step(
                        module,
                        value,
                        labels,
                        torch.tensor(value.numel(), device="cuda"),
                        params,
                        {},
                    )

        assert trace_call.call_count == 1
        assert trainer._traced_step is not None
        targets = {node.target for node in trainer._traced_step.gm.graph.nodes}
        assert torch.ops.torchtitan.accumulate_tensor_statistics.default in targets
        assert (
            torch.ops.torchtitan.record_tensor_statistics_cotangent.default in targets
        )
        assert is_cudagraph_compatible(trainer._traced_step.gm)
        snapshot = runtime.snapshot_unreduced_statistics()
        for public_name in ("hidden", "output.x", "output.dx"):
            assert snapshot[public_name]["counts"].tolist() == [36, 0, 0, 3]
    finally:
        runtime.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_graph_trainer_cudagraph_replay_obeys_device_cadence() -> None:
    torch.manual_seed(0)
    module = TinyStatsModule(width=4, track_forward_calls=False).cuda()
    value = torch.randn(3, 4, device="cuda")
    runtime = init(module, device=torch.device("cuda"))
    try:
        traced = _trace_forward_backward_step(module, value)
        _rematerialize_every_forward_node(traced)
        assert is_cudagraph_compatible(traced.gm)
        traced.gm = cudagraph_pass(traced.gm, traced.example_inputs)
        runner = run_traced(traced, module=module, _validate_runtime=True)

        # Capture may start between logging steps. The device flag must still
        # make later selected replays mutate the captured fixed buffers.
        with set_enabled(False):
            runner(value)  # warmup
            runner(value)  # capture
        captured_snapshot = runtime.snapshot_unreduced_statistics()
        assert all(
            count == 0
            for statistic in captured_snapshot.values()
            for count in statistic["counts"].tolist()
        )

        with set_enabled(True):
            runner(value)
        replay_snapshot = runtime.snapshot_unreduced_statistics()
        assert replay_snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert replay_snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert replay_snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]

        with set_enabled(False):
            runner(value)
        _assert_snapshots_equal(
            runtime.snapshot_unreduced_statistics(),
            replay_snapshot,
        )
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compile_fullgraph_forward_cadence_has_one_stable_graph() -> None:
    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    module = CompileForwardStatsModule(width=4).cuda()
    runtime = init(module, device=torch.device("cuda"))
    try:
        compiled = torch.compile(module, backend=record_graph, fullgraph=True)
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        for enabled in (True, True, False, False, True):
            with set_enabled(enabled):
                compiled(value).sum().backward()

        assert len(compiled_graphs) == 1
        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["hidden"]["counts"].tolist() == [36, 0, 0, 3]
    finally:
        runtime.close()
        torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compile_fullgraph_cadence_keeps_one_forward_and_backward_graph() -> None:
    forward_graphs: list[torch.fx.GraphModule] = []
    backward_graphs: list[torch.fx.GraphModule] = []

    def record_forward(graph_module, _example_inputs):
        forward_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    def record_backward(graph_module, _example_inputs):
        backward_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    torch.compiler.reset()
    counters.clear()
    module = CompileStatsModule(width=4).cuda()
    runtime = init(module, device=torch.device("cuda"))
    try:
        compiled = torch.compile(
            module,
            backend=aot_autograd(
                fw_compiler=record_forward,
                bw_compiler=record_backward,
            ),
            fullgraph=True,
        )
        value = torch.randn(3, 4, device="cuda", requires_grad=True)

        with set_enabled(False):
            compiled(value).sum().backward()
        forward_after_warmup = len(forward_graphs)
        backward_after_warmup = len(backward_graphs)
        unique_graphs_after_warmup = counters["stats"]["unique_graphs"]

        for selected in (True, False, True, False):  # cadence 2
            value.grad = None
            with set_enabled(selected):
                compiled(value).sum().backward()

        assert len(forward_graphs) == forward_after_warmup == 1
        assert len(backward_graphs) == backward_after_warmup == 1
        assert counters["stats"]["unique_graphs"] == unique_graphs_after_warmup == 1
        assert not counters["graph_break"]
        assert "torchtitan.accumulate_tensor_statistics.default" in (
            _graph_argument_names(forward_graphs[0])
        )
        assert "torchtitan.record_tensor_statistics_cotangent.default" in (
            _graph_argument_names(backward_graphs[0])
        )
        snapshot = runtime.snapshot_unreduced_statistics()
        for public_name in ("hidden", "output.x", "output.dx"):
            assert snapshot[public_name]["counts"].tolist() == [24, 0, 0, 2]
    finally:
        runtime.close()
        torch.compiler.reset()
        counters.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compile_reuses_one_graph_across_layers_and_validates_without_grad() -> None:
    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    layers = nn.ModuleList([CompileStatsModule(width=4).cuda() for _ in range(10)])
    layer_sequence = tuple(layers)
    runtime = init(layers, device=torch.device("cuda"))
    try:
        for layer in layer_sequence:
            layer.compile(backend=record_graph, fullgraph=True)

        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            output = value
            for layer in layer_sequence:
                output = layer(output)
            output.sum().backward()

        with torch.no_grad():
            output = value.detach()
            for layer in layer_sequence:
                output = layer(output)

        assert len(compiled_graphs) == 2
        assert all(
            runtime.snapshot_unreduced_statistics()[f"{index}.output.dx"]["counts"][
                3
            ].item()
            == 1
            for index in range(len(layer_sequence))
        )
    finally:
        runtime.close()
        torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compile_fullgraph_records_forward_and_cotangent() -> None:
    module = CompileStatsModule(width=4).cuda()
    runtime = init(module, device=torch.device("cuda"))
    try:
        compiled = torch.compile(module, fullgraph=True)
        values = tuple(
            torch.randn(3, 4, device="cuda", requires_grad=True) for _ in range(2)
        )
        with set_enabled(True):
            outputs = tuple(compiled(value) for value in values)
            sum(output.sum() for output in outputs).backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["hidden"]["counts"].tolist() == [24, 0, 0, 2]
        assert snapshot["output.x"]["counts"].tolist() == [24, 0, 0, 2]
        assert snapshot["output.dx"]["counts"].tolist() == [24, 0, 0, 2]
    finally:
        runtime.close()
        torch.compiler.reset()


@pytest.mark.parametrize(
    "policy",
    [
        FullAC.Config(),
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[]),
    ],
    ids=["full", "selective"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compile_fullgraph_with_ac_records_exactly_once(policy) -> None:
    root = TinyInputStatsRoot().cuda()
    policy.build().apply(root)
    apply_compile(
        root,
        compile_config=CompileConfig(enable=True, components=["model"]),
        parallel_dims=ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        ),
    )
    runtime = init(root, device=torch.device("cuda"))
    try:
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            root(value).sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["layers.0.input.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.input.dx"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()
        torch.compiler.reset()


class TestPipelineMetricReduction(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_unowned_pipeline_row_contributes_reduction_identities(self) -> None:
        owner = nn.Module()
        rank = dist.get_rank()
        register(owner, ["value" if rank == 0 else "other"])
        runtime = tensor_logging_init(
            owner,
            device=torch.device(self.device_type),
            pp_enabled=True,
        )
        try:
            if rank == 0:
                with set_enabled(True):
                    log_stats(
                        owner,
                        value=torch.tensor(
                            [0.0, 1.0, -2.0, 3.0],
                            device=self.device_type,
                        ),
                    )

            assert runtime.collect() == {
                "value.numel": 4,
                "value.nonfinite_count": 0,
                "value.observation_count": 1,
                "value.zero_count": 1,
                "value.zero_frac": 0.25,
                "value.abs_sum": 6.0,
                "value.abs_mean": 1.5,
                "value.square_mean": 3.5,
                "value.rms": 3.5**0.5,
                "value.kurtosis": -1.0,
                "value.abs_max": 3.0,
            }
        finally:
            runtime.close()

    @with_comms
    def test_pipeline_part_layouts_keep_global_metric_names(self) -> None:
        rank = dist.get_rank()
        part_layouts = (
            ((0,), (7,)),
            ((0, 2), (1, 3)),
        )

        for layers_by_rank in part_layouts:
            parts = []
            for layer_id in layers_by_rank[rank]:
                part = nn.Module()
                part.layers = nn.ModuleDict({str(layer_id): nn.Identity()})
                register(part.layers[str(layer_id)], ["hidden"])
                parts.append(part)

            runtime = tensor_logging_init(
                parts,
                device=torch.device(self.device_type),
                pp_enabled=True,
            )
            try:
                expected_names = sorted(
                    f"layers.{layer_id}.hidden"
                    for rank_layers in layers_by_rank
                    for layer_id in rank_layers
                )
                assert runtime.full_metric_names == expected_names
                assert all(
                    not name.startswith("model_parts.") for name in expected_names
                )
            finally:
                runtime.close()
