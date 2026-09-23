# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch

from torchtitan.rl.model import gdn
from torchtitan.rl.model.gdn_backend import (
    GDNExecutionPath,
    TorchTitanGDNAttentionMetadata,
    TorchTitanGDNAttentionMetadataBuilder,
)
from torchtitan.rl.model.vllm_worker import TorchTitanCudagraphDispatcher
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
    builder = TorchTitanGDNAttentionMetadataBuilder(
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
    assert captured.execution_path is GDNExecutionPath.PACKED
    assert captured.num_prefills == 0 and captured.num_decodes == 2
    assert captured.num_decode_tokens == 4 and captured.num_prefill_tokens == 0
    assert captured.has_initial_state is not None
    assert not captured.non_spec_state_indices_tensor.any()
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
    assert decode_metadata.execution_path is GDNExecutionPath.SINGLE_TOKEN
    assert decode_metadata.has_initial_state.tolist() == [True, False]
    assert decode_metadata.num_actual_tokens == 2
    assert decode_metadata.non_spec_query_start_loc.tolist() == [0, 1, 2]
    assert decode_metadata.non_spec_state_indices_tensor.tolist() == [1, 0]
    single_token = replace(common, max_query_len=1, num_actual_tokens=2)
    decode_capture = builder.build_for_cudagraph_capture(single_token)
    assert not decode_capture.has_initial_state.any()
    actual = builder.build(0, single_token)
    assert actual.execution_path is GDNExecutionPath.SINGLE_TOKEN
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
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
    assert dispatcher.dispatch(2, uniform_decode=True)[0] == CUDAGraphMode.NONE


@pytest.mark.parametrize("decode", [False, True])
def test_forward_respects_prepared_extent_and_decode_padding(monkeypatch, decode):
    layer = gdn.VLLMInnerGatedDeltaNet.__new__(gdn.VLLMInnerGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.kv_cache = "gdn", (torch.empty(0), torch.empty(0))
    extent = 4 if decode else 3
    metadata = TorchTitanGDNAttentionMetadata(
        execution_path=(
            GDNExecutionPath.SINGLE_TOKEN if decode else GDNExecutionPath.PACKED
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
    )
    monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: True)

    def convolution(x, *args, has_initial_state=None, **kwargs):
        assert has_initial_state is metadata.has_initial_state
        return x

    monkeypatch.setattr(gdn, "causal_conv1d_decode", convolution)
    monkeypatch.setattr(gdn, "paged_causal_conv1d", convolution)

    def recurrent(conv, a, b, A_log, bias, out, offsets, slots, initial):
        assert conv.shape[0] == out.shape[0] == extent
        assert offsets[-1] == extent
        assert initial is metadata.has_initial_state
        out.fill_(1)

    monkeypatch.setattr(layer, "_forward_gdn", recurrent)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    value, output = torch.ones(6, 1), torch.zeros(6, 1)
    with override_forward_context(context):
        layer._forward(value, value, value, value, None, value, value, output)
        # Reject bias even in profiling and empty-batch early returns.
        for attn_metadata in (None, {"gdn": replace(metadata, num_actual_tokens=0)}):
            context.attn_metadata = attn_metadata
            with pytest.raises(
                AssertionError,
                match="Attention Gym convolution kernels do not support bias",
            ):
                layer._forward(value, value, value, value, value, value, value, output)
    assert metadata.num_actual_tokens == 3
    assert torch.equal(output[:, 0], (torch.arange(6) < extent).float())
