# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    current_platform,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

from torchtitan.experiments.rl.models import gdn as gdn_module
from torchtitan.experiments.rl.models.gdn_backend import (
    TorchTitanGDNAttentionBackend,
    TorchTitanGDNAttentionMetadataBuilder,
)
from torchtitan.experiments.rl.models.gdn_metadata import GDNGraphMetadata
from torchtitan.experiments.rl.models.vllm_worker import (
    TorchTitanGDNGraphWrapper,
    TorchTitanGPUModelRunner,
)


def native_metadata(*, num_decodes=1, offsets=(0, 1, 5), slots=(7, 9), initial=None):
    num_reqs = len(offsets) - 1
    return GDNAttentionMetadata(
        num_prefills=num_reqs - num_decodes,
        num_prefill_tokens=offsets[-1] - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=offsets[-1],
        non_spec_query_start_loc=torch.tensor(offsets, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor(slots, dtype=torch.int32),
        has_initial_state=(
            None if initial is None else torch.tensor(initial, dtype=torch.bool)
        ),
    )


@pytest.fixture
def runner(monkeypatch):
    # Native construction resolves GPU kernels; only the TorchTitan buffers and
    # dispatch are under test. No CUDA allocation/capture is needed here.
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", lambda *args: None)
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: None)
    compilation_config = SimpleNamespace(
        cudagraph_num_of_warmups=1,
        static_forward_context={},
    )
    config = SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    instance = object.__new__(TorchTitanGPUModelRunner)
    instance.vllm_config = config
    instance.compilation_config = compilation_config
    first_builder = TorchTitanGDNAttentionMetadataBuilder(
        None, ["gdn.0", "gdn.1"], config, torch.device("cpu")
    )
    second_builder = TorchTitanGDNAttentionMetadataBuilder(
        None, ["gdn.2"], config, torch.device("cpu")
    )
    instance.attn_groups = [
        [
            AttentionGroup(
                TorchTitanGDNAttentionBackend,
                ["gdn.0", "gdn.1"],
                None,
                0,
                [first_builder],
            ),
            AttentionGroup(object, ["attention"], None, 0, [object()]),
        ],
        [
            AttentionGroup(
                TorchTitanGDNAttentionBackend, ["gdn.2"], None, 1, [second_builder]
            )
        ],
    ]
    return instance


def context(mode, metadata, *, token_capacity=8):
    return ForwardContext(
        no_compile_layers={"layer": object()},
        attn_metadata=metadata,
        slot_mapping={"attention": torch.tensor([3])},
        dp_metadata=object(),
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(num_tokens=token_capacity),
        is_padding=torch.tensor([False, True]),
        additional_kwargs={"sentinel": object()},
    )


@pytest.mark.parametrize(
    "dp_size,use_ubatching,enable_dbo,breakable,enabled",
    [
        (1, False, False, True, True),
        (2, False, False, True, False),
        (1, True, False, True, False),
        (1, False, True, True, False),
        (1, False, False, False, False),
    ],
)
def test_layer_selects_backend_only_for_supported_modes(
    monkeypatch, dp_size, use_ubatching, enable_dbo, breakable, enabled
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=dp_size,
            use_ubatching=use_ubatching,
            enable_dbo=enable_dbo,
        ),
        model_config=object(),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="float32"),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={},
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
    )
    monkeypatch.setattr(gdn_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(gdn_module, "is_breakable_cudagraph_enabled", lambda: breakable)
    monkeypatch.setattr(gdn_module, "is_conv_state_dim_first", lambda: False)
    layer = gdn_module.VLLMInnerGatedDeltaNet(
        gdn_module.VLLMInnerGatedDeltaNet.Config(
            layer_idx=0,
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=128,
            head_v_dim=128,
        )
    )
    assert layer.use_piecewise_capture is enabled
    assert (layer.get_attn_backend() is TorchTitanGDNAttentionBackend) is enabled
    assert layer.mamba_type.name == "GDN_ATTN"


def test_backend_keeps_native_metadata_building(runner):
    builder = runner.attn_groups[0][0].get_metadata_builder()
    assert TorchTitanGDNAttentionBackend.get_builder_cls() is type(builder)
    assert type(builder).build is GDNAttentionMetadataBuilder.build
    assert (
        type(builder).build_for_cudagraph_capture
        is GDNAttentionMetadataBuilder.build_for_cudagraph_capture
    )
    assert (
        type(builder)._cudagraph_support
        == GDNAttentionMetadataBuilder._cudagraph_support
    )
    assert runner.compilation_config.static_forward_context == {}


@pytest.mark.parametrize(
    "num_decodes,offsets,initial,expected_flags",
    [
        (0, (0, 2, 5), (False, True), [False, True]),
        (1, (0, 1, 5), (True, False), [True, False]),
        (2, (0, 1, 2), None, [True, True]),
    ],
)
def test_stage_packed_uses_real_counts_and_clears_tail(
    runner, num_decodes, offsets, initial, expected_flags
):
    builder = runner.attn_groups[0][0].get_metadata_builder()
    metadata = native_metadata(
        num_decodes=num_decodes, offsets=offsets, initial=initial
    )
    # Upstream FULL metadata can have padded rows; only real requests are live.
    metadata.non_spec_query_start_loc = torch.tensor(
        [*offsets, offsets[-1], offsets[-1]], dtype=torch.int32
    )
    metadata.non_spec_state_indices_tensor = torch.tensor([7, 9, 91, 92])
    packed = builder.stage_packed(metadata, token_capacity=8)
    assert packed.query_start_loc.tolist() == [*offsets, 8, 8, 8]
    assert packed.state_indices.tolist() == [7, 9, 0, 0, 0]
    assert packed.has_initial_state.tolist() == [*expected_flags, False, False, False]
    pointers = [getattr(packed, field.name).data_ptr() for field in fields(packed)]

    smaller = native_metadata(offsets=(0, 1), slots=(11,))
    packed = builder.stage_packed(smaller, token_capacity=2)
    assert packed.query_start_loc.tolist() == [0, 1, 2, 2]
    assert packed.state_indices.tolist() == [11, 0, 0]
    assert packed.has_initial_state.tolist() == [True, False, False]
    assert [
        getattr(packed, field.name).data_ptr() for field in fields(packed)
    ] == pointers

    packed = builder.stage_dummy(token_capacity=8)
    assert packed.query_start_loc.tolist() == [0, 8, 8, 8, 8, 8]
    assert packed.state_indices.tolist() == [0] * 5
    assert packed.has_initial_state.tolist() == [False] * 5
    assert [
        getattr(packed, field.name).data_ptr() for field in fields(packed)
    ] == pointers


@pytest.mark.parametrize(
    "mode", sorted(CUDAGraphMode.valid_runtime_modes(), key=lambda mode: mode.name)
)
@pytest.mark.parametrize("dummy", [False, True])
def test_wrapper_routes_and_preserves_context(runner, monkeypatch, mode, dummy):
    first = native_metadata(initial=(True, False))
    second = native_metadata(
        num_decodes=0, offsets=(0, 5), slots=(12,), initial=(False,)
    )
    original = (
        None
        if dummy
        else {
            "gdn.0": first,
            "gdn.1": first,
            "gdn.2": second,
            "attention": object(),
        }
    )
    outer_context = context(mode, original)
    wrapper = TorchTitanGDNGraphWrapper(lambda: None, runner)

    def upstream_call(self, **kwargs):
        prepared = get_forward_context()
        should_stage = mode == CUDAGraphMode.PIECEWISE or (
            mode == CUDAGraphMode.NONE and dummy
        )
        if not should_stage:
            assert prepared is outer_context
            assert prepared.attn_metadata is original
            return "native"

        assert prepared is not outer_context
        for field in fields(outer_context):
            if field.name != "attn_metadata":
                assert getattr(prepared, field.name) is getattr(
                    outer_context, field.name
                )
        metadata = prepared.attn_metadata
        assert metadata["gdn.0"] is metadata["gdn.1"]
        assert metadata["gdn.0"] is not metadata["gdn.2"]
        assert isinstance(metadata["gdn.0"], GDNGraphMetadata)
        assert metadata["attention"] is (None if dummy else original["attention"])
        assert metadata["gdn.0"].state_indices.tolist() == (
            [0] * 5 if dummy else [7, 9, 0, 0, 0]
        )
        assert metadata["gdn.2"].state_indices.tolist() == (
            [0] * 5 if dummy else [12, 0, 0, 0, 0]
        )
        return "packed"

    monkeypatch.setattr(BreakableCUDAGraphWrapper, "__call__", upstream_call)
    with override_forward_context(outer_context):
        result = wrapper(positions=torch.arange(8))
        assert get_forward_context() is outer_context
    assert result in {"native", "packed"}
    if original is not None:
        assert original["gdn.0"] is first
        assert original["gdn.2"] is second


def test_staging_precedes_capture_and_every_replay(runner, monkeypatch):
    wrapper = TorchTitanGDNGraphWrapper(lambda: None, runner)
    observed = []

    def record():
        packed = get_forward_context().attn_metadata["gdn.0"]
        observed.append(
            (packed.query_start_loc.tolist(), packed.state_indices.tolist())
        )

    def capture(self, entry, args, kwargs):
        record()
        entry.capture = object()

    def replay(self, entry, args, kwargs):
        record()

    monkeypatch.setattr(BreakableCUDAGraphWrapper, "_capture", capture)
    monkeypatch.setattr(BreakableCUDAGraphWrapper, "_replay", replay)
    for metadata in [
        native_metadata(initial=(True, False)),
        native_metadata(num_decodes=0, offsets=(0, 2), slots=(4,), initial=(False,)),
        None,
    ]:
        per_layer = (
            None
            if metadata is None
            else {
                "gdn.0": metadata,
                "gdn.1": metadata,
                "gdn.2": metadata,
                "attention": object(),
            }
        )
        with override_forward_context(context(CUDAGraphMode.PIECEWISE, per_layer)):
            wrapper(positions=torch.arange(8))
    assert len(wrapper.entries) == 1
    assert observed == [
        ([0, 1, 5, 8, 8, 8], [7, 9, 0, 0, 0]),
        ([0, 2, 8, 8, 8, 8], [4, 0, 0, 0, 0]),
        ([0, 8, 8, 8, 8, 8], [0, 0, 0, 0, 0]),
    ]


def test_pure_decode_piecewise_preserves_legacy_eager(runner):
    metadata = native_metadata(num_decodes=2, offsets=(0, 1, 2))
    per_layer = {"gdn.0": metadata, "gdn.1": metadata, "gdn.2": metadata}
    original = context(CUDAGraphMode.PIECEWISE, per_layer)

    def run(**kwargs):
        prepared = get_forward_context()
        assert prepared.cudagraph_runtime_mode == CUDAGraphMode.NONE
        assert prepared.attn_metadata is per_layer
        return "legacy"

    wrapper = TorchTitanGDNGraphWrapper(run, runner)
    with override_forward_context(original):
        assert wrapper(positions=torch.arange(8)) == "legacy"
        assert get_forward_context() is original
    assert not wrapper.entries


def test_none_dummy_refreshes_builders_and_uses_positions_capacity(runner):
    wrapper = TorchTitanGDNGraphWrapper(
        lambda **kwargs: get_forward_context().attn_metadata, runner
    )
    outer_context = context(CUDAGraphMode.NONE, None)
    outer_context.batch_descriptor = None
    old_builder = runner.attn_groups[0][0].get_metadata_builder()
    with override_forward_context(outer_context):
        before = wrapper(positions=torch.zeros(3, 2))
        replacement = TorchTitanGDNAttentionMetadataBuilder(
            None, ["gdn.new"], runner.vllm_config, torch.device("cpu")
        )
        runner.attn_groups = [
            [
                AttentionGroup(
                    TorchTitanGDNAttentionBackend, ["gdn.new"], None, 0, [replacement]
                )
            ]
        ]
        after = wrapper(positions=torch.zeros(3, 2))
    assert set(after) == {"gdn.new"}
    assert before["gdn.0"].query_start_loc.tolist() == [0, 2, 2, 2]
    assert after["gdn.new"].query_start_loc.tolist() == [0, 2, 2, 2]
    assert (
        after["gdn.new"].query_start_loc.data_ptr()
        != old_builder.graph_metadata.query_start_loc.data_ptr()
    )


def test_profiling_without_context_or_builders_is_unchanged(runner):
    sentinel = object()
    wrapper = TorchTitanGDNGraphWrapper(lambda: sentinel, runner)
    with override_forward_context(None):
        assert wrapper() is sentinel
    runner.attn_groups = []
    outer_context = context(CUDAGraphMode.NONE, None)
    with override_forward_context(outer_context):
        assert wrapper() is sentinel
        assert get_forward_context().attn_metadata is None


@pytest.mark.parametrize(
    "has_gdn,enabled,warmups",
    [(False, False, 0), (True, False, 0), (True, True, 1), (True, True, 0)],
)
def test_load_model_installs_only_for_enabled_gdn(
    runner, monkeypatch, has_gdn, enabled, warmups
):
    module_name = "torchtitan.experiments.rl.models.gdn"
    if has_gdn:
        layer = gdn_module.VLLMInnerGatedDeltaNet.__new__(
            gdn_module.VLLMInnerGatedDeltaNet
        )
        torch.nn.Module.__init__(layer)
        layer.use_piecewise_capture = enabled
        runner.compilation_config.static_forward_context["gdn"] = layer
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        runner.compilation_config.static_forward_context["attention"] = object()
    runner.compilation_config.cudagraph_num_of_warmups = warmups
    raw_model = lambda: None
    original_wrapper = BreakableCUDAGraphWrapper(raw_model, runner.vllm_config)
    loaded = []

    def load_model(self, load_dummy_weights=False):
        loaded.append(load_dummy_weights)
        self.model = original_wrapper

    monkeypatch.setattr(GPUModelRunner, "load_model", load_model)
    if enabled and warmups == 0:
        with pytest.raises(ValueError, match="cudagraph_num_of_warmups >= 1"):
            runner.load_model(load_dummy_weights=True)
        assert runner.model is original_wrapper
    else:
        runner.load_model(load_dummy_weights=True)
        if enabled:
            assert isinstance(runner.model, TorchTitanGDNGraphWrapper)
            assert runner.model.unwrap() is raw_model
            assert not isinstance(runner.model.unwrap(), BreakableCUDAGraphWrapper)
        else:
            assert runner.model is original_wrapper
    assert loaded == [True]
    if not has_gdn:
        assert module_name not in sys.modules
