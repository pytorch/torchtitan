# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's GDN extend captured in a CUDA graph, on a bucket's
``PrefillGraphMetadata``, against its eager extend (``TorchTitanGDNAttnBackend``).

One bucket is captured once, then replayed for batches of different shapes: many
requests, cached prefixes, padding rows past the batch, prefix-cache snapshots, and
batches without them. Each replay must match the eager extend on the output and on the
whole conv and state pools, the last slot included; only SGLang's dummy slot 0 may
differ. A replay for an idle attention-DP rank (one made-up request, no real row) must
write nothing, for the GDN and the KDA extend. Runs in the B200 RL CI job, with the
SGLang environment (see
torchtitan/rl/generator/sglang/README.md):
    pytest tests/rl/unit_tests/gpu/test_sglang_gdn_prefill_graph.py -v
"""

from types import SimpleNamespace

import pytest
import torch

# The SGLang environment: torchtitan/rl/generator/sglang/README.md.
pytest.importorskip("sglang")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

KEY_HEADS, VALUE_HEADS, HEAD_DIM, CONV_WIDTH = 2, 4, 128, 4
CHANNELS = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM
KDA_HEADS = 2
BUCKET, MAX_SEQS = 512, 160
# Request i's slot is i + 1 and its track slot NUM_SLOTS // 2 + i, so no request owns
# the last slot, where a torch index write of -1 would land.
NUM_SLOTS = 400
CHUNK = 64

# (new tokens, cached prefix lengths, {row: tokens from its prefix to its snapshot}).
# A snapshot sits at a multiple of 64 new tokens; at the row's end, it is the final state.
CASES = [
    ([37, 100, 5, 192], [0, 64, 130, 0], {1: 70, 3: 192}),
    ([BUCKET], [256], {0: 300}),
    (
        [1, 2, 3, 5, 8, 13, 21, 34, 3, 9, 1, 30, 17, 4, 64, 20],
        [0, 5, 0, 64, 0, 3, 0, 128, 0, 0, 7, 0, 0, 1, 0, 64],
        {14: 64},
    ),
    # More requests than SGLang's own GDN backend allows by default (128).
    ([3] * 140 + [64], [0, 9] * 70 + [0], {140: 64}),
    ([200, 11], [0, 0], None),
    ([1], [900], None),
]


def _layer():
    generator = torch.Generator(device="cuda").manual_seed(7)

    def randn(*shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(*shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    key_dim = KEY_HEADS * HEAD_DIM
    return SimpleNamespace(
        layer_id=0,
        num_k_heads=KEY_HEADS,
        num_v_heads=VALUE_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        conv_weights=(
            randn(key_dim, 1, CONV_WIDTH),
            randn(key_dim, 1, CONV_WIDTH),
            randn(VALUE_HEADS * HEAD_DIM, 1, CONV_WIDTH),
        ),
        A_log=randn(VALUE_HEADS, dtype=torch.float32),
        dt_bias=randn(VALUE_HEADS, dtype=torch.float32),
    )


def _backend(
    conv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    kind: str = "gdn",
    replay_pools: tuple[torch.Tensor, ...] = (),
):
    """The production backend (``kind`` "gdn" or "kda") on test pools, with SGLang's
    pool and request table replaced by the parts the extend reads. ``replay_pools`` are
    the KDA slots' replay-state entries (batch-invariant mode)."""
    from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
    from sglang.srt.mem_cache.kv.router import KVLayerRouter, STATE_PART

    from torchtitan.rl.generator.sglang.attention.gdn import TorchTitanGDNAttnBackend
    from torchtitan.rl.generator.sglang.attention.kda import TorchTitanKDAAttnBackend

    backend_class = {"gdn": TorchTitanGDNAttnBackend, "kda": TorchTitanKDAAttnBackend}[
        kind
    ]
    backend = backend_class.__new__(backend_class)
    backend.state_pool = None
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda req_pool_indices: (req_pool_indices + 1).to(
            torch.int32
        )
    )
    backend._state_ids = IdSpace(key=("state", None))
    pool = SimpleNamespace(layer_transfer_counter=None, start_layer=0)
    pool.layer_router = KVLayerRouter.from_layer_maps(pool, ())
    layer_cache = SimpleNamespace(conv=[conv_pool, *replay_pools], temporal=state_pool)
    pool.layer_router.bind_part(
        sub_pool="state",
        part=STATE_PART,
        view=SimpleNamespace(mamba2_layer_cache=lambda layer_id: layer_cache),
        indices={0: 0},
    )
    backend.token_to_kv_pool = pool
    backend.device = torch.device("cuda")
    backend.pad_slot_id = -1
    backend._mamba_chunk_size = CHUNK
    backend.conv_states_shape = conv_pool.transpose(-1, -2).shape
    backend.prefill_graph_max_seqs = MAX_SEQS
    backend.forward_metadata = None
    return backend


def _batch(extend_lens, prefix_lens, track=None):
    """A ``ForwardBatch`` stand-in with the fields the extend reads."""
    from sglang.srt.model_executor.forward_mode import ForwardMode

    batch_size = len(extend_lens)
    starts = [sum(extend_lens[:i]) for i in range(batch_size)]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        logical_forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        # Set by SGLang's attention-DP padding only.
        _original_batch_size=None,
        req_pool_indices=torch.arange(batch_size, device="cuda"),
        out_cache_loc=torch.zeros(sum(extend_lens), dtype=torch.int64, device="cuda"),
        extend_seq_lens=torch.tensor(extend_lens, device="cuda"),
        extend_prefix_lens=torch.tensor(prefix_lens, device="cuda"),
        extend_start_loc=torch.tensor(starts, device="cuda"),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        tbo_parent_token_range=None,
        mamba_track_mask=None,
        mamba_track_indices=None,
        mamba_track_seqlens=None,
        mamba_prefill_track_mask_cpu=None,
        mamba_track_seqlens_cpu=None,
    )
    if track is not None:
        mask = [row in track for row in range(batch_size)]
        track_lens = [prefix_lens[row] + track.get(row, 0) for row in range(batch_size)]
        batch.mamba_prefill_track_mask_cpu = mask
        batch.mamba_track_seqlens_cpu = track_lens
        batch.mamba_track_mask = torch.tensor(mask, device="cuda")
        batch.mamba_track_seqlens = torch.tensor(track_lens, device="cuda")
        batch.mamba_track_indices = torch.arange(
            NUM_SLOTS // 2,
            NUM_SLOTS // 2 + batch_size,
            dtype=torch.int32,
            device="cuda",
        )

    def mamba_track_aligned_lens():
        # ForwardBatch.mamba_track_aligned_lens, with SGLang's cache chunk size of 64.
        if batch.mamba_track_mask is None:
            return None
        return (batch.mamba_track_seqlens - batch.extend_prefix_lens) // CHUNK * CHUNK

    batch.mamba_track_aligned_lens = mamba_track_aligned_lens
    return batch


def _capture_batch():
    """SGLang's capture batch: one request of the bucket's size, with the track slots on
    the device and no CPU track lists (``PrefillCudaGraphRunner.capture_prepare``)."""
    batch = _batch([BUCKET], [0], track={})
    batch.mamba_prefill_track_mask_cpu = None
    batch.mamba_track_seqlens_cpu = None
    return batch


def _idle_rank_batch():
    """SGLang's made-up request for an idle attention-DP rank: one bucket-long request
    on request row 0, and no real row (``_original_batch_size`` 0)."""
    batch = _batch([BUCKET], [0])
    batch._original_batch_size = 0
    return batch


def _capture(extend, pools: list[torch.Tensor]):
    """Capture ``extend`` in a CUDA graph after one warm-up call; ``pools`` end as they
    began. Returns the graph and its output."""
    pools_before_capture = [pool.clone() for pool in pools]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Compiles and autotunes the kernels outside the capture.
        extend()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = extend()
    for pool, pool_before_capture in zip(pools, pools_before_capture, strict=True):
        pool.copy_(pool_before_capture)
    return graph, output


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize(
    "batch_invariant", [False, True], ids=["default", "batch-invariant"]
)
def test_prefill_graph_replays_match_eager_extend(
    batch_invariant: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replays of one captured bucket match the eager extend of each batch.

    Bitwise, except outside batch-invariant mode for a batch without snapshots:
    the graph runs the fused chunk kernel, the eager extend warpdrive.
    """
    monkeypatch.setattr(
        "torchtitan.rl.generator.attention.gdn.is_in_batch_invariant_mode",
        lambda: batch_invariant,
    )

    layer = _layer()
    generator = torch.Generator(device="cuda").manual_seed(11)
    conv_pool = (
        torch.randn(
            NUM_SLOTS, CONV_WIDTH - 1, CHANNELS, generator=generator, device="cuda"
        )
        * 0.3
    ).to(torch.bfloat16)
    state_pool = (
        torch.randn(
            NUM_SLOTS,
            VALUE_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        )
        * 0.05
    )
    backend = _backend(conv_pool, state_pool)

    # The static inputs the graph reads.
    mixed_qkv = torch.zeros(BUCKET, CHANNELS, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(BUCKET, VALUE_HEADS, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(BUCKET, VALUE_HEADS, dtype=torch.bfloat16, device="cuda")

    # ======== Capture one bucket, as SGLang does ========
    capture_batch = _capture_batch()
    metadata = backend.init_prefill_graph_metadata(capture_batch)
    graph, output = _capture(
        lambda: backend.forward_extend(layer, capture_batch, mixed_qkv, a, b),
        [conv_pool, state_pool],
    )

    # ======== Replay other batches against the eager extend ========
    for case, (extend_lens, prefix_lens, track) in enumerate(CASES):
        num_tokens = sum(extend_lens)
        # Garbage past the batch must reach no request.
        mixed_qkv.copy_(
            torch.randn(BUCKET, CHANNELS, generator=generator, device="cuda")
        )
        a.copy_(torch.randn(BUCKET, VALUE_HEADS, generator=generator, device="cuda"))
        b.copy_(torch.randn(BUCKET, VALUE_HEADS, generator=generator, device="cuda"))

        expected_conv_pool, expected_state_pool = conv_pool.clone(), state_pool.clone()
        eager_backend = _backend(expected_conv_pool, expected_state_pool)
        eager_batch = _batch(extend_lens, prefix_lens, track)
        eager_backend.init_forward_metadata(eager_batch)
        expected_output = eager_backend.forward_extend(
            layer,
            eager_batch,
            mixed_qkv[:num_tokens],
            a[:num_tokens],
            b[:num_tokens],
        )

        backend.refresh_prefill_graph_metadata(
            metadata, _batch(extend_lens, prefix_lens, track)
        )
        graph.replay()
        torch.cuda.synchronize()

        name = f"case {case}"
        assert bool((output[:, num_tokens:] == 0).all()), name
        # Both sides convolve with the same kernel and copy the same snapshots; the
        # graph also copies the untracked rows' conv history into the dummy slot 0.
        assert torch.equal(conv_pool[1:], expected_conv_pool[1:]), name
        if batch_invariant or track is not None:
            assert torch.equal(output[:, :num_tokens], expected_output), name
            assert torch.equal(state_pool[1:], expected_state_pool[1:]), name
        else:
            # The requests' slots are 1 to len(extend_lens); no other slot changes.
            request_slots = slice(1, len(extend_lens) + 1)
            assert _rel_l2(output[:, :num_tokens], expected_output) < 1e-2, name
            assert (
                _rel_l2(state_pool[request_slots], expected_state_pool[request_slots])
                < 1e-2
            ), name
            other_slots = slice(len(extend_lens) + 1, None)
            assert torch.equal(
                state_pool[other_slots], expected_state_pool[other_slots]
            ), name

        # The next case starts from the same pools on both sides.
        conv_pool.copy_(expected_conv_pool)
        state_pool.copy_(expected_state_pool)

    # ======== An idle attention-DP rank, after a real batch ========
    # A real batch first, so the idle replay's zero check reads an output with values.
    backend.refresh_prefill_graph_metadata(metadata, _batch([200, 11], [0, 0]))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(output[:, : 200 + 11].any())

    # The stand-in maps request row 0 to slot 1, a real slot: only the tables keep the
    # made-up request off it.
    mixed_qkv.copy_(torch.randn(BUCKET, CHANNELS, generator=generator, device="cuda"))
    pools_before = (conv_pool.clone(), state_pool.clone())
    backend.refresh_prefill_graph_metadata(metadata, _idle_rank_batch())
    graph.replay()
    torch.cuda.synchronize()
    assert bool((output == 0).all())
    assert torch.equal(conv_pool[1:], pools_before[0][1:])
    assert torch.equal(state_pool[1:], pools_before[1][1:])

    # Decode rows run as 1-token extends: the capture batch's track fields, at a replay.
    decode_batch = _batch([1, 1], [40, 70], track={})
    decode_batch.mamba_track_seqlens_cpu = None
    with pytest.raises(RuntimeError, match="no prefill track lengths"):
        backend.refresh_prefill_graph_metadata(metadata, decode_batch)


def _kda_layer():
    generator = torch.Generator(device="cuda").manual_seed(13)

    def randn(*shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(*shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    channels = KDA_HEADS * HEAD_DIM
    return SimpleNamespace(
        layer_id=0,
        head_k_dim=HEAD_DIM,
        conv_weights=tuple(randn(channels, 1, CONV_WIDTH) for _ in range(3)),
        A_log=randn(KDA_HEADS, dtype=torch.float32),
        dt_bias=randn(KDA_HEADS, HEAD_DIM, dtype=torch.float32),
        lower_bound=-5.0,
    )


@pytest.mark.parametrize(
    "batch_invariant", [False, True], ids=["default", "batch-invariant"]
)
def test_kda_idle_rank_replay_writes_nothing(
    batch_invariant: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay of the captured KDA extend for an idle attention-DP rank gives zero
    output and writes no slot but SGLang's dummy slot 0, replay state included."""
    from torchtitan.rl.generator.attention.kda import (
        REPLAY_STATE_DTYPES,
        replay_state_shapes,
    )
    from torchtitan.rl.generator.sglang.attention.kda import _flat_shape_in_dtype

    monkeypatch.setattr(
        "torchtitan.rl.generator.sglang.attention.kda.is_in_batch_invariant_mode",
        lambda: batch_invariant,
    )

    layer = _kda_layer()
    generator = torch.Generator(device="cuda").manual_seed(17)
    channels = KDA_HEADS * HEAD_DIM
    conv_pool = (
        torch.randn(
            NUM_SLOTS, CONV_WIDTH - 1, 3 * channels, generator=generator, device="cuda"
        )
        * 0.3
    ).to(torch.bfloat16)
    state_pool = (
        torch.randn(
            NUM_SLOTS, KDA_HEADS, HEAD_DIM, HEAD_DIM, generator=generator, device="cuda"
        )
        * 0.05
    )
    # Batch-invariant mode keeps Attention Gym's replay state in extra conv entries.
    replay_pools = ()
    if batch_invariant:
        replay_pools = tuple(
            torch.zeros(
                NUM_SLOTS,
                *_flat_shape_in_dtype(shape, dtype, torch.bfloat16),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for shape, dtype in zip(
                replay_state_shapes(KDA_HEADS, HEAD_DIM),
                REPLAY_STATE_DTYPES,
                strict=True,
            )
        )
    backend = _backend(conv_pool, state_pool, "kda", replay_pools)
    pools = [conv_pool, state_pool, *replay_pools]

    # The static inputs the graph reads.
    mixed_qkv = torch.zeros(BUCKET, 3 * channels, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(BUCKET, channels, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(BUCKET, KDA_HEADS, dtype=torch.bfloat16, device="cuda")

    capture_batch = _capture_batch()
    metadata = backend.init_prefill_graph_metadata(capture_batch)
    graph, output = _capture(
        lambda: backend.forward_extend(layer, capture_batch, mixed_qkv, a, b), pools
    )

    def replay(batch) -> list[torch.Tensor]:
        """Replay ``batch`` on random inputs; returns the pools as they were before."""
        mixed_qkv.copy_(
            torch.randn(BUCKET, 3 * channels, generator=generator, device="cuda")
        )
        a.copy_(torch.randn(BUCKET, channels, generator=generator, device="cuda"))
        b.copy_(torch.randn(BUCKET, KDA_HEADS, generator=generator, device="cuda"))
        pools_before = [pool.clone() for pool in pools]
        backend.refresh_prefill_graph_metadata(metadata, batch)
        graph.replay()
        torch.cuda.synchronize()
        return pools_before

    # A real batch first, so the idle replay's zero check reads an output with values.
    replay(_batch([37, 100], [0, 64]))
    assert bool(output[:, : 37 + 100].any())

    pools_before = replay(_idle_rank_batch())
    assert bool((output == 0).all())
    for pool, pool_before in zip(pools, pools_before, strict=True):
        # The replay entries keep float32 and int32 bytes in bfloat16 storage, where
        # some bit patterns read as NaN: compare the bits.
        assert torch.equal(
            pool[1:].view(torch.uint8), pool_before[1:].view(torch.uint8)
        )
