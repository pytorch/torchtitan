# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import context_parallel
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    LinearAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
)
from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attention.cp_linear_attention import (
    prepare_linear_attention_cp_metadata,
)
from torchtitan.models.common.attn_gym import ContextParallelRouting, is_missing_kernel
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding

pytestmark = pytest.mark.skipif(
    is_missing_kernel(ContextParallelRouting), reason="Attention Gym is not installed"
)


def _assert_same_routing(actual, expected):
    for name, value in vars(expected).items():
        result = getattr(actual, name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(result, value), name
        else:
            assert result == value, name


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("headtail", [False, True])
@pytest.mark.parametrize("lengths", [[16], [3, 6, 7], [1, 1, 2, 4, 8]])
def test_cpu_layout_matches_device_offsets(cp_size, headtail, lengths):
    positions = torch.cat([torch.arange(length) for length in lengths])
    # Include padded offset slots as used by CUDA-graph input preprocessing.
    varlen = create_varlen_metadata_for_document(
        positions, max_num_documents=8, max_context_length=16
    )
    metadata = LinearAttentionMetadata(varlen=varlen, num_conv_history_tokens=3)
    permutation = (
        context_parallel._HeadTailLoadBalancer(16, cp_size, "cpu")._generate_indices()
        if headtail
        else None
    )
    fragments = (
        context_parallel.PTRRFlexAttentionCPLoadBalancer.get_token_fragments(
            mock.Mock(spec=context_parallel.PTRRFlexAttentionCPLoadBalancer),
            permutation,
            cp_size=cp_size,
        )
        if permutation is not None
        else [
            [(rank * (16 // cp_size), (rank + 1) * (16 // cp_size))]
            for rank in range(cp_size)
        ]
    )
    cu_seqlens = tuple((positions == 0).nonzero().flatten().tolist()) + (16,)
    for rank in range(cp_size):
        reference = ContextParallelRouting.from_fragments(
            varlen.cu_seq_q.tolist(), fragments, rank, "cpu", conv_history=3
        )
        with mock.patch(
            "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
            return_value=SimpleNamespace(size=lambda: cp_size),
        ), mock.patch(
            "torchtitan.models.common.attention.cp_linear_attention.dist.get_rank",
            return_value=rank,
        ):
            result = prepare_linear_attention_cp_metadata(
                metadata,
                permutation=permutation,
                cu_seqlens=cu_seqlens,
                fragments=tuple(tuple(row) for row in fragments),
                family="KDA",
            )
        _assert_same_routing(result.cp_routing, reference)


class _NoToListTensor(torch.Tensor):
    def tolist(self):
        raise AssertionError("Device inputs and metadata must not be read back")


def _token_microbatch(positions, **model_kwargs):
    return TokenizedTrainingMicrobatch(
        input=torch.arange(positions.numel()),
        labels=torch.arange(positions.numel()),
        positions=positions,
        padding_mask=torch.zeros(positions.numel(), dtype=torch.bool),
        loss_token_counts=torch.tensor(positions.numel()),
        routing_token_counts=torch.tensor([positions.numel()]),
        model_kwargs=model_kwargs,
    )


@pytest.mark.parametrize("lengths", [(8,), (3, 5), (1, 1, 2, 4)])
def test_batch_retains_document_boundaries_before_upload(lengths):
    positions = torch.cat([torch.arange(length) for length in lengths])
    # MRoPE coordinates differ from the 1D document positions.
    microbatch = _token_microbatch(positions, mrope_positions=torch.ones(8, 3))
    batch = microbatch.to_input_dict("meta", include_cu_seqlens=True)
    expected = (0, *torch.tensor(lengths).cumsum(0).tolist())
    assert batch["cu_seqlens"] == expected
    assert batch["positions"].device.type == "meta"
    assert batch["mrope_positions"].device.type == "meta"
    assert microbatch.positions.device.type == "cpu"
    assert "cu_seqlens" not in microbatch.as_input_dict()


def test_batch_skips_document_boundaries_unless_requested():
    microbatch = _token_microbatch(torch.arange(8).as_subclass(_NoToListTensor))
    assert "cu_seqlens" not in microbatch.to_input_dict("meta")


def test_batch_boundaries_require_cpu_positions():
    microbatch = _token_microbatch(torch.arange(8, device="meta"))
    with pytest.raises(ValueError, match="1D CPU positions"):
        microbatch.to_input_dict("meta", include_cu_seqlens=True)


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("headtail", [False, True])
@pytest.mark.parametrize(
    "backends",
    [
        (ContextParallelInnerKDA,),
        (ContextParallelInnerGDN,),
        (ContextParallelInnerKDA, ContextParallelInnerGDN),
    ],
)
def test_preprocessing_uses_batch_boundaries_without_device_reads(
    mrope, headtail, backends
):
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    offsets = torch.tensor([0, 3, 8, 8, 8], dtype=torch.int32).as_subclass(
        _NoToListTensor
    )
    metadata = LinearAttentionMetadata(
        varlen=VarlenAttentionMetadata(offsets, offsets, 5, 5),
        num_conv_history_tokens=3,
    )
    batch = _token_microbatch(positions).to_input_dict("cpu", include_cu_seqlens=True)
    batch["positions"] = (
        positions[:, None].expand(-1, 3) if mrope else positions
    ).as_subclass(_NoToListTensor)
    batch["attention_metadata"] = {backend: metadata for backend in backends}
    model = SimpleNamespace(
        config=SimpleNamespace(
            first_full_attention=SimpleNamespace(
                inner_attention=AllGatherKVFlexInnerAttention.Config()
            )
        )
    )
    with mock.patch(
        "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
        return_value=SimpleNamespace(size=lambda: 2),
    ), mock.patch(
        "torchtitan.models.common.attention.cp_linear_attention.dist.get_rank",
        return_value=1,
    ), mock.patch.object(
        context_parallel, "shard_tensors", side_effect=lambda x, **_: x
    ), mock.patch.object(
        context_parallel,
        "spmd_mesh_group",
        return_value=SimpleNamespace(size=lambda: 2),
    ), mock.patch.object(
        context_parallel,
        "current_spmd_mesh",
        return_value=SimpleNamespace(device_type="cpu"),
    ):
        result = Decoder._cp_shard(
            model,
            batch,
            input_shardings=decoder_input_sharding(),
            parallelism_context=SimpleNamespace(cp=2),
            parallelism=ParallelismConfig(
                context_parallel_degree=2,
                context_parallel_load_balancer=(
                    context_parallel.HeadTailCPLoadBalancer.Config()
                    if headtail
                    else None
                ),
            ),
        )
    reference = ContextParallelRouting.from_fragments(
        [0, 3, 8],
        [[(0, 2), (6, 8)], [(2, 6)]] if headtail else [[(0, 4)], [(4, 8)]],
        1,
        "cpu",
        conv_history=3,
    )
    for backend in backends:
        _assert_same_routing(
            result["attention_metadata"][backend].cp_routing, reference
        )
    assert "cu_seqlens" not in result


def test_preprocessing_requires_batch_boundaries_for_linear_attention():
    batch = {
        "input": torch.arange(8),
        "attention_metadata": {
            ContextParallelInnerKDA: LinearAttentionMetadata(
                varlen=None, num_conv_history_tokens=3
            )
        },
    }
    with pytest.raises(ValueError, match="include_cu_seqlens=True"):
        Decoder._cp_shard(
            SimpleNamespace(),
            batch,
            input_shardings=decoder_input_sharding(),
            parallelism_context=SimpleNamespace(cp=2),
            parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        )


@pytest.mark.parametrize("backend", [ContextParallelInnerKDA, ContextParallelInnerGDN])
@pytest.mark.parametrize(
    "cu_seqlens,fragments",
    [(None, None), (None, (((0, 4),), ((4, 8),))), ((0, 8), None)],
)
def test_linear_attention_requires_cpu_boundaries_and_fragments(
    backend, cu_seqlens, fragments
):
    metadata = LinearAttentionMetadata(
        varlen=create_varlen_metadata_for_document(torch.arange(8)),
        num_conv_history_tokens=3,
    )
    with pytest.raises(ValueError, match="requires CPU document boundaries"):
        backend.prepare_cp_metadata(
            metadata, permutation=None, cu_seqlens=cu_seqlens, fragments=fragments
        )


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_headtail_fragments_do_not_read_device_permutation(cp_size, chunk_size):
    num_tokens = 2 * cp_size * chunk_size
    permutation = context_parallel._HeadTailLoadBalancer(
        num_tokens, cp_size, "cpu"
    )._generate_indices()
    expected = context_parallel.PTRRFlexAttentionCPLoadBalancer.get_token_fragments(
        mock.Mock(spec=context_parallel.PTRRFlexAttentionCPLoadBalancer),
        permutation,
        cp_size=cp_size,
    )
    with mock.patch.object(
        context_parallel,
        "spmd_mesh_group",
        return_value=SimpleNamespace(size=lambda: cp_size),
    ), mock.patch.object(
        context_parallel,
        "current_spmd_mesh",
        return_value=SimpleNamespace(device_type="cpu"),
    ):
        balancer = context_parallel.HeadTailCPLoadBalancer.Config().build(
            seq_len=num_tokens, attention_metadata=None
        )
    actual = balancer.get_token_fragments(
        permutation.as_subclass(_NoToListTensor), cp_size=cp_size
    )
    assert actual == expected
