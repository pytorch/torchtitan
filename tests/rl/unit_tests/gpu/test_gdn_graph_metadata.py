# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, replace

import pytest
import torch

from torchtitan.rl.generator.vllm import gdn
from torchtitan.rl.generator.vllm.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionMetadata,
    TorchTitanLinearAttentionMetadataBuilder,
)
from torchtitan.rl.generator.vllm.worker import TorchTitanCudagraphDispatcher
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


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
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        builder.build(0, oversized_common)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        builder.build_for_cudagraph_capture(oversized_common)

    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
    assert dispatcher.dispatch(2, uniform_decode=True)[0] == CUDAGraphMode.NONE


@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize("single_token", [False, True])
def test_update_block_table_matches_build(single_token, mode):
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
    offsets = torch.tensor([0, 1, 2] if single_token else [0, 1, 3], dtype=torch.int32)
    num_tokens = int(offsets[-1])
    common = CommonAttentionMetadata(
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        # "align": running states in block-table columns 1 and 0; "none": column 0.
        seq_lens=torch.tensor([16, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=num_tokens,
        max_query_len=1 if single_token else 2,
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


@pytest.mark.parametrize("decode", [False, True])
def test_forward_respects_prepared_extent_and_decode_padding(monkeypatch, decode):
    layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.kv_cache = "gdn", (torch.empty(0), torch.empty(0))
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
    layer.local_num_k_heads = layer.local_num_v_heads = 1
    layer.head_k_dim = layer.head_v_dim = layer.local_key_dim = 1
    monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(gdn, "paged_causal_conv1d", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(gdn, "gate_transform", lambda a, *args, **kwargs: a)
    monkeypatch.setattr(gdn, "l2norm", lambda x, **kwargs: x)

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
    monkeypatch.setattr(gdn, "paged_chunk_gdn", chunk)
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
    layer.local_num_k_heads = layer.local_num_v_heads = 16
    layer.head_k_dim = layer.head_v_dim = 128
    layer.local_key_dim = 2048
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
        layer.kv_cache = tuple(pool.clone() for pool in pools)
        with torch.inference_mode(), override_forward_context(context):
            output = layer._forward(inputs, a, b, weight, None, A_log, dt_bias)
        return output.float(), layer.kv_cache[1][1:]

    (routed, routed_state), (reference, reference_state) = run(False), run(True)
    for got, want in ((routed, reference), (routed_state, reference_state)):
        assert ((got - want).norm() / want.norm()).item() < 1e-2
