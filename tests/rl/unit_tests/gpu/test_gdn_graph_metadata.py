# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from torchtitan.rl.generator.attention import gdn as paged_gdn
from torchtitan.rl.generator.vllm.attention import gdn, linear_attention
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionBackend,
    TorchTitanLinearAttentionMetadata,
    TorchTitanLinearAttentionMetadataBuilder,
)
from torchtitan.rl.generator.vllm.worker import (
    TorchTitanCudagraphDispatcher,
    TorchTitanGPUModelRunner,
)
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@pytest.mark.parametrize(
    "graph_mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY],
)
def test_gdn_supports_eager_and_full_decode_only_modes(graph_mode):
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=graph_mode,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=2,
        cudagraph_num_of_warmups=1,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=16,
        max_num_seqs=2,
        max_num_batched_tokens=2,
        is_encoder_decoder=False,
    )
    spec = MambaSpec(
        block_size=8,
        shapes=((3, 512), (2, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )

    TorchTitanLinearAttentionMetadataBuilder(spec, [], config, torch.device("cpu"))

    dispatcher = TorchTitanCudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(graph_mode)
    decode_mode, _ = dispatcher.dispatch(1, uniform_decode=True)
    prefill_mode, _ = dispatcher.dispatch(2, uniform_decode=False)
    expected_decode = (
        CUDAGraphMode.FULL
        if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        else CUDAGraphMode.NONE
    )
    assert (decode_mode, prefill_mode) == (expected_decode, CUDAGraphMode.NONE)


@pytest.mark.parametrize(
    "data_parallel_size,batch_invariant,expected",
    [
        (2, False, AttentionCGSupport.ALWAYS),
        (2, True, AttentionCGSupport.UNIFORM_BATCH),
        (1, True, AttentionCGSupport.ALWAYS),
    ],
)
def test_dp_support_level(monkeypatch, data_parallel_size, batch_invariant, expected):
    config = VllmConfig()
    config.parallel_config.data_parallel_size = data_parallel_size
    monkeypatch.setattr(
        linear_attention, "is_in_batch_invariant_mode", lambda: batch_invariant
    )
    support = TorchTitanLinearAttentionMetadataBuilder.get_cudagraph_support(
        config, None
    )
    assert support == expected


def test_full_metadata_and_native_dispatch_variants():
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        cudagraph_num_of_warmups=1,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=16,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        is_encoder_decoder=False,
    )
    config.cache_config.mamba_cache_mode = "align"
    dispatcher = TorchTitanCudagraphDispatcher(config)
    assert dispatcher.dispatch(2, uniform_decode=True)[0] == CUDAGraphMode.NONE
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)
    decode_mode, decode = dispatcher.dispatch(2, uniform_decode=True)
    packed_mode, packed = dispatcher.dispatch(2)
    assert decode_mode == packed_mode == CUDAGraphMode.FULL
    assert decode != packed and decode.uniform and not packed.uniform
    for mode, descriptors in dispatcher.get_capture_descs():
        for descriptor in descriptors:
            assert dispatcher.dispatch(
                descriptor.num_tokens, uniform_decode=descriptor.uniform
            ) == (mode, descriptor)
    assert dispatcher.dispatch(9)[0] == CUDAGraphMode.NONE
    assert (
        dispatcher.dispatch(2, uniform_decode=True, valid_modes={CUDAGraphMode.NONE})[0]
        == CUDAGraphMode.NONE
    )

    spec = MambaSpec(
        block_size=8,
        shapes=((3, 512), (2, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    builder = TorchTitanLinearAttentionMetadataBuilder(
        spec, [], config, torch.device("cpu")
    )
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([1, 3], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[1], [2]], dtype=torch.int32),
        slot_mapping=torch.zeros(4, dtype=torch.int64),
    )
    captured = builder.build_for_cudagraph_capture(common)
    assert isinstance(captured, GDNAttentionMetadata)
    # Native splits stay truthful; the graph key keeps this general dummy packed.
    assert captured.execution_path is LinearAttentionExecutionPath.PACKED
    assert captured.num_prefills == 0 and captured.num_decodes == 2
    assert captured.num_decode_tokens == 4 and captured.num_prefill_tokens == 0
    assert captured.has_initial_state is not None
    assert not captured.non_spec_state_indices_tensor.any()
    assert not captured.decode_state_indices.any()
    assert not captured.prefill_state_indices.any()
    assert captured.spec_sequence_masks is None and captured.chunk_indices is None
    assert common.block_table_tensor.tolist() == [[1], [2]]
    actual = builder.build(0, common)
    for name in (
        "non_spec_query_start_loc",
        "non_spec_state_indices_tensor",
        "has_initial_state",
    ):
        assert getattr(captured, name).data_ptr() == getattr(actual, name).data_ptr()
    assert actual.non_spec_query_start_loc.tolist() == [0, 1, 2, 4]
    assert actual.non_spec_state_indices_tensor.tolist() == [1, 2, 0]
    assert actual.has_initial_state.tolist() == [False, True, False]
    assert actual.num_actual_tokens == 4
    padded_offsets = torch.tensor([0, 1, 1], dtype=torch.int32)
    decode_metadata = builder.build(
        0,
        replace(
            common,
            max_query_len=1,
            num_actual_tokens=2,
            query_start_loc=padded_offsets,
            query_start_loc_cpu=padded_offsets,
            seq_lens=torch.tensor([3, 0], dtype=torch.int32),
            block_table_tensor=torch.tensor([[1], [0]], dtype=torch.int32),
            _num_computed_tokens_cache=None,
        ),
    )
    assert decode_metadata.execution_path is LinearAttentionExecutionPath.SINGLE_TOKEN
    assert decode_metadata.has_initial_state.tolist() == [True, False]
    assert decode_metadata.num_actual_tokens == 2
    assert decode_metadata.non_spec_query_start_loc.tolist() == [0, 1, 2]
    assert decode_metadata.non_spec_state_indices_tensor.tolist() == [1, 0]
    single_token = replace(common, max_query_len=1, num_actual_tokens=2)
    decode_capture = builder.build_for_cudagraph_capture(single_token)
    # A one-token FULL dummy (2 <= max_num_seqs) can replay the mixed-batch graph,
    # so capture restages the decode split: 2, not the stale 4 from above.
    assert builder.prefill_query_start_loc.tolist() == [0, 2, 2, 2]
    assert not decode_capture.has_initial_state.any()
    actual = builder.build(0, single_token)
    assert actual.execution_path is LinearAttentionExecutionPath.SINGLE_TOKEN
    assert actual.has_initial_state.tolist() == [False, True]
    for name in (
        "non_spec_query_start_loc",
        "non_spec_state_indices_tensor",
        "has_initial_state",
    ):
        assert getattr(decode_capture, name).shape == getattr(actual, name).shape
        assert (
            getattr(decode_capture, name).data_ptr() == getattr(actual, name).data_ptr()
        )
    assert actual.num_prefills == 0 and actual.num_decodes == 2

    # vLLM can synthesize one sequence as large as the FULL capture token
    # bucket. Reject a bucket larger than the GDN model context before its state
    # block table reaches an out-of-bounds gather.
    oversized_offsets = torch.tensor([0, 32], dtype=torch.int32)
    oversized_common = replace(
        common,
        query_start_loc=oversized_offsets,
        query_start_loc_cpu=oversized_offsets,
        seq_lens=torch.tensor([32], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=32,
        max_query_len=32,
        max_seq_len=32,
        block_table_tensor=torch.tensor([[1]], dtype=torch.int32),
        slot_mapping=torch.zeros(32, dtype=torch.int64),
        _num_computed_tokens_cache=None,
    )
    with pytest.raises(ValueError, match="block-table capacity"):
        builder.build(0, oversized_common)
    with pytest.raises(ValueError, match="block-table capacity"):
        builder.build_for_cudagraph_capture(oversized_common)

    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
    assert dispatcher.dispatch(2, uniform_decode=True)[0] == CUDAGraphMode.NONE


@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize(
    "step",
    ["packed", "single_token", "padded_single_token", "packed_graph_single_token"],
)
def test_update_block_table_matches_build(step, mode):
    """vLLM reuses one GDN group's metadata for the next; only the state slots differ."""
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        cudagraph_num_of_warmups=1,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=16,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        is_encoder_decoder=False,
    )
    config.cache_config.mamba_cache_mode = mode
    spec = MambaSpec(
        block_size=8,
        shapes=((3, 512), (2, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    first, other, reference = (
        TorchTitanLinearAttentionMetadataBuilder(spec, [], config, torch.device("cpu"))
        for _ in range(3)
    )
    replays_packed_graph = step in ("padded_single_token", "packed_graph_single_token")
    for builder in (first, other, reference):
        builder.replays_packed_graph = replays_packed_graph
    offsets = torch.tensor(
        [0, 1, 3] if step == "packed" else [0, 1, 2], dtype=torch.int32
    )
    # Padded past max_num_seqs (2), a one-token step replays the mixed-batch graph.
    num_tokens = 4 if step == "padded_single_token" else int(offsets[-1])
    common = CommonAttentionMetadata(
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        # "align": running states in block-table columns 1 and 0; "none": column 0.
        seq_lens=torch.tensor([16, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=num_tokens,
        max_query_len=2 if step == "packed" else 1,
        max_seq_len=16,
        block_table_tensor=torch.tensor([[1, 5], [2, 6]], dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )
    other_table = torch.tensor([[3, 7], [4, 8]], dtype=torch.int32)
    captured = other.build_for_cudagraph_capture(common)
    # Stale values from an earlier step, so every copy and zeroing is checked.
    for buffer in (
        other.query_start_loc,
        other.state_indices,
        other.has_initial_state,
        other.decode_state_indices,
        other.prefill_query_start_loc,
        other.prefill_state_indices,
    ):
        buffer.fill_(True if buffer.dtype == torch.bool else -1)

    updated = other.update_block_table(
        first.build(0, common), other_table, common.slot_mapping
    )

    expected = reference.build(0, replace(common, block_table_tensor=other_table))
    assert expected.non_spec_state_indices_tensor.tolist()[:2] == (
        [7, 4] if mode == "align" else [3, 4]
    )
    for field in fields(expected):
        want, got = getattr(expected, field.name), getattr(updated, field.name)
        if not isinstance(want, torch.Tensor):
            assert got == want, field.name
            continue
        assert torch.equal(got, want), field.name
        # The graph captured this builder's buffers, not the first group's.
        if field.name != "state_block_column":
            assert (
                got.data_ptr() == getattr(captured, field.name).data_ptr()
            ), field.name
    if replays_packed_graph:
        # The mixed-batch graph reads this builder's offsets.
        assert other.query_start_loc.tolist() == [0, 1, 2, num_tokens]


@pytest.mark.parametrize(
    "mode, uniform, replays_packed_graph",
    [
        (CUDAGraphMode.FULL, False, True),
        (CUDAGraphMode.FULL, True, False),
        (CUDAGraphMode.NONE, False, False),
    ],
)
def test_runner_tells_builders_the_selected_graph(mode, uniform, replays_packed_graph):
    builder, other_builder = SimpleNamespace(), SimpleNamespace()
    runner = cast(Any, object.__new__(TorchTitanGPUModelRunner))
    runner.get_model = lambda: SimpleNamespace(set_num_valid_tokens=lambda n: None)
    runner.attn_groups = [
        [
            SimpleNamespace(
                backend=TorchTitanLinearAttentionBackend, metadata_builders=[builder]
            ),
            SimpleNamespace(backend=object, metadata_builders=[other_builder]),
        ]
    ]
    selected = (mode, BatchDescriptor(4, uniform=uniform))
    with patch.object(
        GPUModelRunner, "_determine_batch_execution_and_padding", return_value=selected
    ):
        runner._determine_batch_execution_and_padding(4, 4)
    assert builder.replays_packed_graph is replays_packed_graph
    assert not hasattr(other_builder, "replays_packed_graph")


@pytest.mark.parametrize("decode", [False, True])
def test_forward_respects_prepared_extent_and_decode_padding(monkeypatch, decode):
    layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.kv_cache = "gdn", (torch.empty(0), torch.empty(0))
    layer.num_speculative_tokens = 0
    layer.local_num_v_heads = layer.head_v_dim = 1
    extent = 4 if decode else 3
    metadata = TorchTitanLinearAttentionMetadata(
        execution_path=(
            LinearAttentionExecutionPath.SINGLE_TOKEN
            if decode
            else LinearAttentionExecutionPath.PACKED
        ),
        # General capture dummies can have zero native prefills as well.
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=extent,
        num_decode_tokens=extent,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc=(
            torch.arange(5) if decode else torch.tensor([0, 1, 2, 3, 3])
        ),
        non_spec_state_indices_tensor=(
            torch.tensor([1, 2, 0, 0]) if decode else torch.tensor([1, 2, 3, 0])
        ),
        has_initial_state=torch.tensor([False, True, False, False]),
        state_block_column=torch.zeros(extent, 1, dtype=torch.long),
    )
    monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: True)

    def convolution(x, *args, has_initial_state=None, **kwargs):
        assert has_initial_state is metadata.has_initial_state
        return x

    monkeypatch.setattr(gdn, "causal_conv1d_decode", convolution)
    monkeypatch.setattr(gdn, "paged_causal_conv1d", convolution)

    def recurrent(conv, a, b, A_log, bias, offsets, slots, initial):
        assert conv.shape[0] == extent
        assert offsets[-1] == extent
        assert initial is metadata.has_initial_state
        return torch.ones(extent, 1, 1)

    monkeypatch.setattr(layer, "_forward_gdn", recurrent)
    # NaN catches a row that is neither written nor zeroed.
    monkeypatch.setattr(
        torch.Tensor, "new_empty", lambda self, size: self.new_full(size, float("nan"))
    )
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    value = torch.ones(6, 1)
    with override_forward_context(context):
        output = layer._forward(value, value, value, value, None, value, value)
        # Profiling and empty-batch early returns reject bias and return zeros.
        for attn_metadata in (None, {"gdn": replace(metadata, num_actual_tokens=0)}):
            context.attn_metadata = attn_metadata
            with pytest.raises(
                AssertionError,
                match="Attention Gym convolution kernels do not support bias",
            ):
                layer._forward(value, value, value, value, value, value, value)
            early = layer._forward(value, value, value, value, None, value, value)
            assert torch.equal(early, torch.zeros(6, 1, 1))
    assert metadata.num_actual_tokens == 3
    assert torch.equal(output.flatten(), (torch.arange(6) < extent).float())


def _packed_metadata(
    offsets: list[int],
    seq_lens: list[int],
    slots: list[int],
    max_num_seqs: int,
    device: str = "cpu",
) -> TorchTitanLinearAttentionMetadata:
    """Real "none"-mode builder metadata for a packed batch, one state slot per request."""
    config = VllmConfig()
    config.scheduler_config = SchedulerConfig(
        max_model_len=128,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=128,
        is_encoder_decoder=False,
    )
    config.cache_config.mamba_cache_mode = "none"
    spec = MambaSpec(
        block_size=8,
        shapes=((3, 512), (2, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    builder = TorchTitanLinearAttentionMetadataBuilder(
        spec, [], config, torch.device(device)
    )
    query_start_loc = torch.tensor(offsets, dtype=torch.int32)
    block_table = torch.tensor(slots, device=device, dtype=torch.int32).unsqueeze(1)
    return builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=query_start_loc.to(device),
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
            num_reqs=len(seq_lens),
            num_actual_tokens=offsets[-1],
            max_query_len=int(query_start_loc.diff().max()),
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=torch.zeros(offsets[-1], device=device, dtype=torch.int64),
        ),
    )


@pytest.mark.parametrize("capturing", [False, True])
def test_packed_forward_routes_decode_rows(monkeypatch, capturing):
    """Decode rows take the decode kernel, prefill rows the chunk kernel; the add keeps both.

    Graph capture runs the decode kernel on every request row; eager, on the decode requests.
    """
    # `_stage_decode_split`'s example: 2 decode requests, then a 3-token prefill; 6 rows.
    metadata = _packed_metadata([0, 1, 2, 5], [6, 4, 3], [5, 7, 9], max_num_seqs=4)
    assert metadata.decode_state_indices.tolist() == [5, 7, 0, 0]
    assert metadata.prefill_query_start_loc.tolist() == [0, 2, 2, 5, 5, 5]
    assert metadata.prefill_state_indices.tolist() == [0, 0, 9, 0, 0]
    layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.kv_cache = "gdn", (torch.empty(0), torch.empty(0))
    layer.num_speculative_tokens = 0
    layer.local_num_k_heads = layer.local_num_v_heads = 1
    layer.head_k_dim = layer.head_v_dim = 1
    monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: False)
    monkeypatch.setattr(paged_gdn, "is_in_batch_invariant_mode", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(gdn, "paged_causal_conv1d", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(paged_gdn, "gate_transform", lambda a, *args, **kwargs: a)
    monkeypatch.setattr(paged_gdn, "l2norm", lambda x, **kwargs: x)

    def decode(conv, a, b, A_log, dt_bias, ssm, slots, *, out, **kwargs):
        rows = 4 if capturing else 2
        assert conv.shape[0] == rows
        assert slots.tolist() == metadata.decode_state_indices[:rows].tolist()
        out.copy_((slots > 0).float()[None, :, None, None])  # inactive rows are zero

    def chunk(q, k, v, decay, beta, ssm, slots, *, cu_seqlens, **kwargs):
        assert cu_seqlens is metadata.prefill_query_start_loc
        assert slots is metadata.prefill_state_indices
        tokens = torch.arange(q.shape[1])
        # The null-slot sequence (the decode tokens, [0, 2)) is zero.
        return torch.where(tokens >= 2, 2.0, 0.0)[None, :, None, None]

    monkeypatch.setattr(gdn, "recurrent_gdn_decode", decode)
    monkeypatch.setattr(paged_gdn, "paged_chunk_gdn", chunk)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    mixed, ab = torch.ones(6, 3), torch.ones(6, 1)
    with override_forward_context(context):
        output = layer._forward(mixed, ab, ab, torch.ones(3, 4), None, ab[0], ab[0])
    assert output.flatten().tolist() == [1, 1, 2, 2, 2, 0]
    if not capturing:
        # No decode requests: eager skips the decode kernel, which rejects 0 rows.
        context.attn_metadata = {"gdn": replace(metadata, num_decodes=0)}
        monkeypatch.setattr(gdn, "recurrent_gdn_decode", lambda *a, **k: pytest.fail())
        with override_forward_context(context):
            output = layer._forward(mixed, ab, ab, torch.ones(3, 4), None, ab[0], ab[0])
        assert output.flatten().tolist() == [0, 0, 2, 2, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_decode_routing_matches_recurrence(monkeypatch) -> None:
    """A mixed step's decode rows (decode kernel) and prefill rows (chunk kernel) match the
    batch-invariant path, which runs the plain recurrence over the same packed batch."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Qwen3.5-0.8B's local TP=1 shapes: 3 resumed decode requests, then a fresh 61-token prefill.
    num_decodes, prefill = 3, 61
    num_tokens = num_decodes + prefill
    metadata = _packed_metadata(
        [0, 1, 2, 3, num_tokens],
        [40, 40, 40, prefill],
        [1, 2, 3, 4],
        max_num_seqs=8,
        device="cuda",
    )
    assert metadata.num_decodes == num_decodes
    layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.num_speculative_tokens = 0
    layer.local_num_k_heads = layer.local_num_v_heads = 16
    layer.head_k_dim = layer.head_v_dim = 128
    pools = (
        (torch.randn(5, 3, 6144, device=device) * 0.5).bfloat16(),
        torch.randn(5, 16, 128, 128, device=device) * 0.05,
    )
    inputs = torch.randn(num_tokens, 6144, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(6144, 4, device=device) / 2).bfloat16()
    a, b = torch.randn(2, num_tokens, 16, device=device, dtype=torch.bfloat16)
    A_log = torch.log1p(torch.rand(16, device=device) * 15)
    dt_bias = torch.full((16,), -3.0, device=device)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )

    def run(batch_invariant: bool) -> tuple[torch.Tensor, torch.Tensor]:
        monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: batch_invariant)
        monkeypatch.setattr(
            paged_gdn, "is_in_batch_invariant_mode", lambda: batch_invariant
        )
        layer.kv_cache = tuple(pool.clone() for pool in pools)
        with torch.inference_mode(), override_forward_context(context):
            output = layer._forward(inputs, a, b, weight, None, A_log, dt_bias)
        return output.float(), layer.kv_cache[1][1:]

    (routed, routed_state), (reference, reference_state) = run(False), run(True)
    for got, want in ((routed, reference), (routed_state, reference_state)):
        assert ((got - want).norm() / want.norm()).item() < 1e-2


# Qwen3.5-0.8B's local TP=1 GDN shapes: 16 heads of 128, q | k | v channels, width-4 conv.
_HEADS, _HEAD_DIM = 16, 128
_CHANNELS = 3 * _HEADS * _HEAD_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_accepted", [1, 2, 4])
def test_verify_rows_match_single_token_decodes(num_accepted) -> None:
    """Verifying 1 + k tokens, j = `num_accepted` of them accepted, matches j single-token decodes.

        draft step:  verify x0 x1 x2 x3 from state S        (keeps one state per token)
        next step:   verify y0 y1 y2 y3, num_accepted_tokens = j
        reference:   decode x0 .. x(j-1), then y0 .. y3, one token per step, no drafts

    The draft step's first j outputs, and the states it keeps after x(j-1), match the
    reference: the rejected drafts change nothing. The next step starts from those states.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = _random_gdn_weights(device)
    # State S: the conv's last 3 inputs and the SSM state.
    history = (torch.randn(3, _CHANNELS, device=device) * 0.5).bfloat16()
    state = torch.randn(_HEADS, _HEAD_DIM, _HEAD_DIM, device=device) * 0.05
    # The verify request owns slots 1-4, one per draft position. Its conv state is in slot 1.
    verify_layer = _qwen3_5_gdn_layer(num_speculative_tokens=3, num_slots=5)
    verify_layer.kv_cache[0][1, :3] = history
    verify_layer.kv_cache[1][1] = state
    decode_layer = _qwen3_5_gdn_layer(num_speculative_tokens=0, num_slots=2)
    decode_layer.kv_cache[0][1] = history
    decode_layer.kv_cache[1][1] = state
    draft_tokens, next_tokens = _random_tokens(4, device), _random_tokens(4, device)

    def verify(tokens, context_len, num_accepted_tokens):
        metadata = _speculative_metadata(
            query_lens=[4],
            context_lens=[context_len],
            slots=[[1, 2, 3, 4]],
            num_draft_tokens=[3],
            num_accepted_tokens=[num_accepted_tokens],
        )
        return _run(verify_layer, metadata, tokens, weights)

    def decode(tokens, context_len):
        outputs = []
        for index in range(tokens[0].shape[0]):
            metadata = _packed_metadata(
                [0, 1], [context_len + index + 1], [1], max_num_seqs=1, device="cuda"
            )
            one_token = tuple(tensor[index : index + 1] for tensor in tokens)
            outputs.append(_run(decode_layer, metadata, one_token, weights))
        return torch.cat(outputs)

    draft_output = verify(draft_tokens, context_len=40, num_accepted_tokens=1)
    accepted_tokens = tuple(tensor[:num_accepted] for tensor in draft_tokens)
    expected = decode(accepted_tokens, context_len=40)
    _assert_close(draft_output[:num_accepted], expected)
    # Kept for the next step: the SSM state after x(j-1) in slot j, and the last 3 inputs in
    # conv rows j - 1 to j + 1.
    _assert_close(verify_layer.kv_cache[1][num_accepted], decode_layer.kv_cache[1][1])
    assert torch.equal(
        verify_layer.kv_cache[0][1, num_accepted - 1 : num_accepted + 2],
        decode_layer.kv_cache[0][1],
    )
    next_output = verify(
        next_tokens, context_len=40 + num_accepted, num_accepted_tokens=num_accepted
    )
    _assert_close(next_output, decode(next_tokens, context_len=40 + num_accepted))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kinds",
    [("decode", "verify", "prefill"), ("decode", "prefill"), ("decode",)],
    ids="+".join,
)
def test_speculative_step_matches_step_without_drafts(kinds) -> None:
    """Each row of a step with drafts on matches the same request in a step without drafts.

        decode, verify, prefill:  vLLM's builder runs the decode row as a 1-token prefill
        decode, prefill:          no verify rows; the decode kernel runs the decode row
        decode:                   a decode-only step

    Requests come in vLLM's order: up to 1 + k tokens first, then prefills. The reference
    step runs the verify row's 4 tokens as one chunk. Outputs, SSM states and conv states
    must match.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = _random_gdn_weights(device)
    # kind: (query_len, context_len, num_draft_tokens, num_accepted_tokens)
    shapes = {
        "decode": (1, 40, -1, 1),
        "verify": (4, 40, 3, 2),
        "prefill": (13, 0, -1, 1),
    }
    query_lens, context_lens, num_draft_tokens, num_accepted_tokens = zip(
        *(shapes[kind] for kind in kinds)
    )
    # Request i owns slots 4i + 1 to 4i + 4 with drafts, slot i + 1 without.
    slots = [
        [4 * index + column for column in range(1, 5)] for index in range(len(kinds))
    ]
    drafts_layer = _qwen3_5_gdn_layer(
        num_speculative_tokens=3, num_slots=4 * len(kinds) + 1
    )
    plain_layer = _qwen3_5_gdn_layer(num_speculative_tokens=0, num_slots=len(kinds) + 1)
    for index, accepted in enumerate(num_accepted_tokens):
        history = (torch.randn(3, _CHANNELS, device=device) * 0.5).bfloat16()
        state = torch.randn(_HEADS, _HEAD_DIM, _HEAD_DIM, device=device) * 0.05
        # A row starts from what its last step kept after `accepted` tokens (1 without drafts).
        drafts_layer.kv_cache[0][slots[index][0], accepted - 1 : accepted + 2] = history
        drafts_layer.kv_cache[1][slots[index][accepted - 1]] = state
        plain_layer.kv_cache[0][index + 1] = history
        plain_layer.kv_cache[1][index + 1] = state
    tokens = _random_tokens(sum(query_lens), device)

    drafts_metadata = _speculative_metadata(
        query_lens=list(query_lens),
        context_lens=list(context_lens),
        slots=slots,
        num_draft_tokens=list(num_draft_tokens),
        num_accepted_tokens=list(num_accepted_tokens),
    )
    offsets = torch.tensor([0, *query_lens]).cumsum(0).tolist()
    plain_metadata = _packed_metadata(
        offsets,
        [context + query for context, query in zip(context_lens, query_lens)],
        list(range(1, len(kinds) + 1)),
        max_num_seqs=4,
        device="cuda",
    )
    drafts_output = _run(drafts_layer, drafts_metadata, tokens, weights)
    plain_output = _run(plain_layer, plain_metadata, tokens, weights)
    for index, kind in enumerate(kinds):
        rows = slice(offsets[index], offsets[index + 1])
        _assert_close(drafts_output[rows], plain_output[rows])
        # A verify row keeps the state after its 4th token in its 4th slot, and its last 3
        # inputs in conv rows 3-5, where the next step reads them if all 4 are accepted.
        last_column = 3 if kind == "verify" else 0
        _assert_close(
            drafts_layer.kv_cache[1][slots[index][last_column]],
            plain_layer.kv_cache[1][index + 1],
        )
        assert torch.equal(
            drafts_layer.kv_cache[0][slots[index][0], last_column : last_column + 3],
            plain_layer.kv_cache[0][index + 1],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_speculative_step_replays_in_breakable_graph(monkeypatch) -> None:
    """A breakable piecewise graph captured without metadata runs the GDN layer eagerly at
    replay, with the replay step's metadata (vLLM captures piecewise graphs without any).

        capture:  piece 1 | eager GDN (no metadata: zeros) | piece 2: output * 2
        replay:   piece 1 | eager GDN (decode + verify + prefill step) | piece 2
    """
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = _random_gdn_weights(device)
    conv_weight, A_log, dt_bias = weights
    query_lens, context_lens = [1, 4, 13], [40, 40, 0]
    slots = [[4 * index + column for column in range(1, 5)] for index in range(3)]
    metadata = _speculative_metadata(
        query_lens=query_lens,
        context_lens=context_lens,
        slots=slots,
        num_draft_tokens=[-1, 3, -1],
        num_accepted_tokens=[1, 2, 1],
    )
    tokens = _random_tokens(sum(query_lens), device)
    eager_layer = _qwen3_5_gdn_layer(num_speculative_tokens=3, num_slots=13)
    graph_layer = _qwen3_5_gdn_layer(num_speculative_tokens=3, num_slots=13)
    graph_layer.kv_cache = tuple(state.clone() for state in eager_layer.kv_cache)
    static_tokens = tuple(torch.zeros_like(tensor) for tensor in tokens)

    no_metadata = ForwardContext(
        no_compile_layers={}, attn_metadata=None, slot_mapping={}
    )
    capture = BreakableCUDAGraphCapture()
    with (
        torch.inference_mode(),
        override_forward_context(no_metadata),
        torch.cuda.stream(torch.cuda.Stream()),
    ):
        with capture:
            doubled = 2 * graph_layer._forward(
                *static_tokens, conv_weight, None, A_log, dt_bias
            )
    assert capture.num_eager_breaks == 1

    for static, tensor in zip(static_tokens, tokens):
        static.copy_(tensor)
    step = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    with torch.inference_mode(), override_forward_context(step):
        capture.replay()
    torch.cuda.synchronize()

    expected = _run(eager_layer, metadata, tokens, weights)
    _assert_close(doubled.float() / 2, expected)
    for graph_state, eager_state in zip(graph_layer.kv_cache, eager_layer.kv_cache):
        assert torch.equal(graph_state, eager_state)


def _qwen3_5_gdn_layer(num_speculative_tokens: int, num_slots: int) -> gdn.VLLMInnerGDN:
    """A `VLLMInnerGDN` with random conv and SSM pools; slot 0 is vLLM's null slot.

    The conv pool keeps 3 + k inputs per slot (width - 1 + k), as vLLM allocates it.
    """
    layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.num_speculative_tokens = num_speculative_tokens
    layer.local_num_k_heads = layer.local_num_v_heads = _HEADS
    layer.head_k_dim = layer.head_v_dim = _HEAD_DIM
    conv_rows = 3 + num_speculative_tokens
    layer.kv_cache = (
        (torch.randn(num_slots, conv_rows, _CHANNELS, device="cuda") * 0.5).bfloat16(),
        torch.randn(num_slots, _HEADS, _HEAD_DIM, _HEAD_DIM, device="cuda") * 0.05,
    )
    return layer


def _random_gdn_weights(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Random (conv_weight, A_log, dt_bias)."""
    conv_weight = (torch.randn(_CHANNELS, 4, device=device) / 2).bfloat16()
    A_log = torch.log1p(torch.rand(_HEADS, device=device) * 15)
    dt_bias = torch.full((_HEADS,), -3.0, device=device)
    return conv_weight, A_log, dt_bias


def _random_tokens(num_tokens: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """(mixed_qkv, a, b) for `num_tokens` tokens."""
    mixed_qkv = torch.randn(num_tokens, _CHANNELS, device=device, dtype=torch.bfloat16)
    a, b = torch.randn(2, num_tokens, _HEADS, device=device, dtype=torch.bfloat16)
    return mixed_qkv, a, b


def _run(layer, metadata, tokens, weights) -> torch.Tensor:
    conv_weight, A_log, dt_bias = weights
    mixed_qkv, a, b = tokens
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    with torch.inference_mode(), override_forward_context(context):
        # vLLM's conv update writes its output over its input.
        output = layer._forward(
            mixed_qkv.clone(), a, b, conv_weight, None, A_log, dt_bias
        )
    return output.float()


def _assert_close(got: torch.Tensor, want: torch.Tensor) -> None:
    error = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert error < 1e-2, error


def _speculative_metadata(
    query_lens: list[int],
    context_lens: list[int],
    slots: list[list[int]],
    num_draft_tokens: list[int],
    num_accepted_tokens: list[int],
) -> GDNAttentionMetadata:
    """vLLM's GDN builder metadata for one step with k = 3 drafts.

    Each request lists 1 + k state slots: a verify row (3 draft tokens) uses all of them,
    other rows (-1) only the first. `num_accepted_tokens` counts the last step's accepted tokens.
    """
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=3, parallel_drafting=False
    )
    config.model_config = SimpleNamespace(hf_text_config=SimpleNamespace())
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[4, 8, 16],
        max_cudagraph_capture_size=16,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        is_encoder_decoder=False,
    )
    config.cache_config.mamba_cache_mode = "none"
    spec = MambaSpec(
        block_size=8,
        shapes=((6, _CHANNELS), (_HEADS, _HEAD_DIM, _HEAD_DIM)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    builder = GDNAttentionMetadataBuilder(spec, [], config, torch.device("cuda"))
    query_start_loc = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0).int()
    seq_lens = [context + query for context, query in zip(context_lens, query_lens)]
    num_tokens = sum(query_lens)
    return builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=query_start_loc.cuda(),
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
            num_reqs=len(query_lens),
            num_actual_tokens=num_tokens,
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=torch.tensor(slots, device="cuda", dtype=torch.int32),
            slot_mapping=torch.zeros(num_tokens, device="cuda", dtype=torch.int64),
        ),
        num_accepted_tokens=torch.tensor(
            num_accepted_tokens, device="cuda", dtype=torch.int32
        ),
        num_decode_draft_tokens_cpu=torch.tensor(num_draft_tokens, dtype=torch.int32),
    )
