# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from torchtitan.distributed import utils as distributed_utils
from torchtitan.experiments.rl.models.gdn import VLLMInnerGatedDeltaNet
from torchtitan.experiments.rl.models.gdn_backend import (
    TorchTitanGDNAttentionMetadataBuilder,
)
from torchtitan.experiments.rl.models.vllm_worker import TorchTitanGDNGraphWrapper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_gdn_graph_replay_matches_eager(monkeypatch, batch_invariant):
    monkeypatch.setattr(distributed_utils, "_batch_invariant_enabled", batch_invariant)
    torch.manual_seed(42)
    tokens, key_heads, value_heads, dim, width = 192, 1, 2, 128, 4
    channels = (2 * key_heads + value_heads) * dim
    layer = VLLMInnerGatedDeltaNet.__new__(VLLMInnerGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.prefix = "probe.linear_attn"
    layer.head_k_dim = layer.head_v_dim = dim
    layer.local_num_k_heads = key_heads
    layer.local_num_v_heads = value_heads
    layer.local_key_dim = key_heads * dim
    layer.conv_kernel_size = width
    # Match the slot-strided layout of vLLM's packed conv/SSM cache pages.
    conv_storage = torch.randn(
        7, (width - 1) * channels + 16, device="cuda", dtype=torch.bfloat16
    )
    ssm_storage = torch.randn(7, value_heads * dim * dim + 16, device="cuda")
    layer.kv_cache = (
        conv_storage[:, : (width - 1) * channels].view(7, width - 1, channels),
        ssm_storage[:, : value_heads * dim * dim].view(7, value_heads, dim, dim),
    )
    q = torch.randn(tokens, key_heads * dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(tokens, value_heads * dim, device="cuda", dtype=q.dtype)
    a = torch.randn(tokens, value_heads, device="cuda", dtype=q.dtype)
    b = torch.randn_like(a)
    weights = tuple(
        torch.randn(c, 1, width, device="cuda", dtype=q.dtype) * 0.1
        for c in (key_heads * dim, key_heads * dim, value_heads * dim)
    )
    A_log = torch.randn(value_heads, device="cuda")
    dt_bias = torch.randn_like(A_log)
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    def run():
        return layer(
            q,
            k,
            v,
            a,
            b,
            *weights,
            A_log,
            dt_bias,
            offsets,
            key_head_dim=dim,
            value_head_dim=dim,
        )

    storage = (conv_storage, ssm_storage)
    compilation = SimpleNamespace(
        cudagraph_num_of_warmups=1,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        max_cudagraph_capture_size=192,
        static_forward_context={layer.prefix: layer},
    )
    config = SimpleNamespace(
        compilation_config=compilation,
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=128)
        ),
        additional_config={"gdn_prefill_backend": "triton"},
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    spec = MambaSpec(
        block_size=128,
        shapes=tuple(tuple(cache.shape[1:]) for cache in layer.kv_cache),
        dtypes=tuple(cache.dtype for cache in layer.kv_cache),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    builder = TorchTitanGDNAttentionMetadataBuilder(
        spec, [layer.prefix], config, torch.device("cuda")
    )
    group = SimpleNamespace(
        layer_names=[layer.prefix], get_metadata_builder=lambda: builder
    )
    runner = SimpleNamespace(
        compilation_config=compilation, vllm_config=config, attn_groups=[[group]]
    )
    wrapper = TorchTitanGDNGraphWrapper(lambda **kwargs: run(), runner)
    # Do not reuse vLLM's process-global pool across independent test lifetimes.
    wrapper.graph_pool = torch.cuda.graph_pool_handle()
    positions = torch.arange(tokens, device="cuda")
    descriptor = BatchDescriptor(num_tokens=tokens)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=descriptor,
    )
    with torch.no_grad(), override_forward_context(context):
        for _ in range(2):
            wrapper(positions=positions)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        with torch.cuda.stream(stream):
            wrapper(positions=positions)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert wrapper.entries[descriptor].capture.num_graphs == 1
        assert wrapper.entries[descriptor].capture.num_eager_breaks == 0

        for iteration, lengths in enumerate(
            ([127], [31, 96], [1, 31, 95], [1, 1, 1], [65]) * 2
        ):
            slots = ([5, 1, 3] if iteration % 2 == 0 else [2, 6, 4])[: len(lengths)]
            offsets_cpu = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
            computed = torch.tensor(
                [
                    64 if length == 1 or (index + iteration) % 2 else 0
                    for index, length in enumerate(lengths)
                ],
                dtype=torch.int32,
            )
            seq_lens_cpu = torch.tensor(lengths, dtype=torch.int32) + computed
            common = CommonAttentionMetadata(
                query_start_loc=offsets_cpu.cuda(),
                query_start_loc_cpu=offsets_cpu,
                seq_lens=seq_lens_cpu.cuda(),
                num_reqs=len(lengths),
                num_actual_tokens=sum(lengths),
                max_query_len=max(lengths),
                max_seq_len=int(seq_lens_cpu.max()),
                block_table_tensor=torch.tensor(
                    slots, device="cuda", dtype=torch.int32
                )[:, None],
                slot_mapping=torch.zeros(
                    sum(lengths), device="cuda", dtype=torch.int64
                ),
            )
            metadata = builder.build(0, common)
            context.attn_metadata = {layer.prefix: metadata}
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            v.mul_(0.9)
            before = tuple(tensor.clone() for tensor in storage)
            expected = run().clone()
            expected_storage = tuple(tensor.clone() for tensor in storage)
            for tensor, snapshot in zip(storage, before, strict=True):
                tensor.copy_(snapshot)
            context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            actual = wrapper(positions=positions)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for tensor, reference in zip(storage, expected_storage, strict=True):
                torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        assert len(wrapper.entries) == 1
