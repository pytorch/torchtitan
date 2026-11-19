# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

pytest.importorskip(
    "attn_gym.sparse.paged_decode",
    reason="Flash-Next cache tests require Attention Gym paged sparse attention",
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.embedding import NGramTable, PLELayer
from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)
from torchtitan.rl.generator.local_compile import INFERENCE_BATCH_INVARIANT_REGION
from torchtitan.rl.generator.vllm import ple
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionMetadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PLE cache requires a GPU"
)


_PLE_REGIONS = ["offset_rmsnorm", "ple_hash_ids", "ple_gate", "ple_conv"]


@pytest.mark.parametrize(
    "regions",
    [
        [],
        ["ple_conv", INFERENCE_BATCH_INVARIANT_REGION],
        _PLE_REGIONS + [INFERENCE_BATCH_INVARIANT_REGION],
    ],
)
def test_ple_chunked_history_reordering_eos_and_page_reuse(monkeypatch, regions):
    """Split PLE execution must match full requests even after reordering and slot reuse."""
    LocalCompileConfig(regions=regions).apply()
    try:
        _check_chunked_history(monkeypatch)
    finally:
        LocalCompileConfig().apply()


def _check_chunked_history(monkeypatch):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, mamba_cache_mode="none"
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(ple, "get_current_vllm_config", lambda: config)
    context = {}
    monkeypatch.setattr(
        ple,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=context, dp_metadata=None),
    )
    torch.manual_seed(42)
    ple_config = (
        build_model_config("debugmodel", seq_len=64, parallelism=ParallelismConfig())
        .layers[1]
        .ple
    )
    ple_config.ple_embedding.eos_token_id = 63
    reference = ple_config.build().cuda()
    reference.init_states()
    for parameter in reference.parameters():
        parameter.data = parameter.data.bfloat16()
    cached = (
        ple.VLLMPLELayer.Config(
            **{
                field.name: getattr(ple_config, field.name)
                for field in fields(ple_config)
            },
            state_metadata_prefix="model.layers.1.linear_attn",
        )
        .build()
        .cuda()
    )
    for parameter in cached.parameters():
        parameter.data = parameter.data.bfloat16()
    cached.init_states()
    cached.load_state_dict(reference.state_dict(), strict=True)
    cached.requires_grad_(False)
    channels = ple_config.dim * ple_config.num_streams
    config.compilation_config.static_forward_context[
        cached.state_metadata_prefix
    ] = SimpleNamespace(kv_cache=(torch.empty(6, 1, device="cuda"),))
    cached.allocate_request_states()
    assert cached.conv_history_BWC is not None and cached.id_history_BI is not None
    cached.conv_history_BWC.copy_(torch.randn_like(cached.conv_history_BWC))
    cached.id_history_BI.random_(0, 248320)
    ids = [torch.randint(0, 248320, (27,), device="cuda") for _ in range(3)]
    ids[0][1] = 0x7FC1
    ids[0][2] = ids[0][15] = 63
    inputs = [
        torch.randn(27, channels, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    expected = [
        reference(
            x.clone().requires_grad_(),
            input_ids_T=tokens,
            document_positions_T=torch.arange(27, device="cuda"),
        ).detach()
        for x, tokens in zip(inputs, ids)
    ]
    completed = [0, 0, 0]
    # Request 2 reuses request 0's state slot after it completes.
    slots = [4, 3, 4]
    for step in (
        [(0, 3), (1, 11)],
        [(1, 1), (0, 14)],
        [(0, 9), (1, 14)],
        [(0, 1), (1, 1)],
        [(2, 1)],
        [(2, 26)],
    ):
        offsets = [0]
        positions, tokens, x = [], [], []
        for request, count in step:
            start = completed[request]
            positions.append(torch.arange(start, start + count, device="cuda"))
            tokens.append(ids[request][start : start + count])
            x.append(inputs[request][start : start + count])
            offsets.append(offsets[-1] + count)
        states = [slots[request] for request, _ in step]
        single_token = all(count == 1 for _, count in step)
        if single_token:
            padding = 4 - len(step)
            positions.append(torch.zeros(padding, device="cuda", dtype=torch.long))
            tokens.append(torch.randint(0, 248320, (padding,), device="cuda"))
            x.append(
                torch.randn(padding, channels, device="cuda", dtype=torch.bfloat16)
            )
            offsets = list(range(5))
            states += [0] * padding
        context[cached.state_metadata_prefix] = TorchTitanLinearAttentionMetadata(
            execution_path=(
                LinearAttentionExecutionPath.SINGLE_TOKEN
                if single_token
                else LinearAttentionExecutionPath.PACKED
            ),
            num_prefills=0 if single_token else len(step),
            num_prefill_tokens=0 if single_token else offsets[-1],
            num_decodes=len(step) if single_token else 0,
            num_decode_tokens=len(step) if single_token else 0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=offsets[-1],
            non_spec_query_start_loc=torch.tensor(
                offsets, device="cuda", dtype=torch.int32
            ),
            non_spec_state_indices_tensor=torch.tensor(
                states, device="cuda", dtype=torch.int32
            ),
            state_block_column=torch.zeros(len(step), 1, device="cuda").long(),
        )
        with torch.inference_mode():
            actual = cached(
                torch.cat(x),
                input_ids_T=torch.cat(tokens),
                document_positions_T=torch.cat(positions),
            )
        for row, (request, count) in enumerate(step):
            start = completed[request]
            torch.testing.assert_close(
                actual[offsets[row] : offsets[row + 1]],
                expected[request][start : start + count],
                atol=0,
                rtol=0,
            )
            completed[request] += count


def test_generator_gate_matches_trainer_training_gate():
    """The generator's split gate regions must round like the trainer's training graph."""
    ple_config = next(
        iter(
            build_model_config(
                "125B-A6B-text", parallelism=ParallelismConfig()
            ).traverse(PLELayer.Config)
        )
    )[1]
    with torch.device("meta"):
        layer = ple_config.build()
    torch.manual_seed(0)
    for norm in (layer.norm_key, layer.norm_query, layer.norm_conv):
        norm.to_empty(device="cuda")
        norm.weight.data = torch.randn_like(norm.weight, dtype=torch.bfloat16) * 0.1
    channels = layer.dim * layer.num_streams
    LocalCompileConfig(regions=["offset_rmsnorm", "ple_gate"]).apply()
    try:
        for num_tokens in (8, 64, 1744):
            key = torch.randn(num_tokens, channels, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(num_tokens, channels, device="cuda", dtype=torch.bfloat16)
            value = torch.randn(num_tokens, layer.dim, device="cuda").bfloat16()
            expected = layer.gate(
                key.clone().requires_grad_(), x.clone().requires_grad_(), value
            )
            with torch.inference_mode():
                actual = ple.VLLMPLELayer._decode_gate(
                    layer, ple.VLLMPLELayer._decode_score(layer, key, x), value
                )
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e.detach(), atol=0, rtol=0)
    finally:
        LocalCompileConfig().apply()


class TestGatherOwnedRows(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_matches_routed_lookup_with_uneven_tokens(self):
        torch.manual_seed(42)
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("dp",))
        num_rows, num_heads = 64, 4
        full_RE = torch.randn(num_rows, 8, device="cuda", dtype=torch.bfloat16)
        full_RE[::3, ::2] = -0.0
        rows_per_owner = num_rows // self.world_size
        start = self.rank * rows_per_owner
        table = NGramTable(NGramTable.Config(num_embeddings=num_rows, embedding_dim=8))
        table.row_mesh = mesh
        table.weight = torch.nn.Parameter(
            DTensor.from_local(
                full_RE[start : start + rows_per_owner].clone(), mesh, [Shard(0)]
            ),
            requires_grad=False,
        )
        num_tokens = 5
        hashes_TN = torch.randint(
            0, num_rows, (num_tokens - 2 * self.rank, num_heads), device="cuda"
        )
        hashes_TN[0, 1] = hashes_TN[0, 0]
        routing = prepare_row_sharded_embedding(
            hashes_TN,
            num_embeddings=num_rows,
            row_mesh=mesh,
            max_ids_per_owner=num_tokens * num_heads,
        )
        expected = row_sharded_embedding(cast(DTensor, table.weight), routing)
        try:
            for regions in ([], [INFERENCE_BATCH_INVARIANT_REGION]):
                LocalCompileConfig(regions=regions).apply()
                actual = ple.gather_owned_rows(
                    table, hashes_TN, num_tokens=num_tokens, dtype=torch.bfloat16
                )
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        finally:
            LocalCompileConfig().apply()
