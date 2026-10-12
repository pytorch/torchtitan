# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SessionKVHolder on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align" groups,
16-token blocks standing in for Qwen3.5's 1,152."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from torchtitan.rl.session_kv_holder import SessionKVHolder
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import make_offload_key, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.request import Request, RequestStatus

BLOCK = 16


@pytest.fixture
def manager() -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    gdn = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    groups = [KVCacheGroupSpec(["attn"], attention)] + [
        KVCacheGroupSpec([f"gdn{i}"], gdn) for i in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=80,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )


def _request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=10_000),
        pooling_params=None,
        cache_salt="v0",
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _schedule(manager, request, num_tokens, blocks=None, hit=0, sample=None) -> None:
    manager.new_step_starts()
    assert manager.allocate_slots(
        request, num_tokens, num_new_computed_tokens=hit, new_computed_blocks=blocks
    )
    request.num_computed_tokens += hit + num_tokens
    if sample is not None:
        request.append_output_token_ids(sample)


def _prefill(manager, request) -> int:
    """Block-aligned chunk, then the tail (like vLLM's Mamba-aligned split); returns the hit."""
    manager.new_step_starts()
    blocks, hit, _ = manager.get_computed_blocks(request)
    num_prompt = request.num_prompt_tokens
    aligned = num_prompt - num_prompt % BLOCK
    first_hit = hit
    if aligned > hit:
        _schedule(manager, request, aligned - hit, blocks, hit)
        blocks, hit = None, 0
    _schedule(
        manager, request, num_prompt - request.num_computed_tokens - hit, blocks, hit, 7
    )
    return first_hit


def _churn(manager, tag: str) -> None:
    """Push every block through the free queue with throwaway requests."""
    for i in range(80):
        request = _request(f"{tag}{i}", [1000 + 97 * i + j for j in range(BLOCK + 1)])
        _prefill(manager, request)
        manager.free(request)


def _holder(
    manager,
    requests,
    floor=0,
    running=(),
    waiting=(),
    skipped_waiting=(),
    hold=True,
    offload_policy=None,
) -> SessionKVHolder:
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        max_num_scheduled_tokens=0,
        requests=requests,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped_waiting,
    )
    holder = SessionKVHolder(scheduler, hold=hold, offload_policy=offload_policy)
    holder.free_floor_blocks = floor
    return holder


def test_held_prefix_survives_churn_and_the_next_turn_hits_it(manager):
    requests = {}
    holder = _holder(manager, requests)
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn1)
    holder.after_step()
    # Decode moves the GDN checkpoint out of the request; other requests churn the free queue.
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    _churn(manager, "a")
    manager.free(turn1)
    del requests["t1"]
    _churn(manager, "b")

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    # The continuation resumes at turn 1's last full prompt block (48 of 50 tokens).
    assert _prefill(manager, turn2) == 48


def test_without_holding_the_next_turn_recomputes(manager):
    turn1 = _request("t1", list(range(50)))
    _prefill(manager, turn1)
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    _churn(manager, "a")
    manager.free(turn1)
    _churn(manager, "b")
    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    assert _prefill(manager, turn2) == 0


def test_release_returns_every_held_block(manager):
    requests = {}
    holder = _holder(manager, requests)
    free_before = manager.block_pool.get_num_free_blocks()
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=3)
    _prefill(manager, turn1)
    holder.after_step()
    manager.free(turn1)
    assert manager.block_pool.get_num_free_blocks() < free_before

    holder.release(group_ids=[3])
    assert holder.num_sessions == 0
    assert manager.block_pool.get_num_free_blocks() == free_before


def _finished_turn(
    manager, holder, requests, session: str, tokens: list[int], group_id: int = 0
) -> Request:
    """One turn of `session`: admit, prefill (held), finish."""
    request = _request(session, tokens)
    requests[session] = request
    holder.track(session, session_id=session, group_id=group_id)
    _prefill(manager, request)
    holder.after_step()
    manager.free(request)
    del requests[session]
    holder.after_step()
    return request


def test_release_frees_sessions_in_sorted_order(manager):
    """Every TP rank frees a release's sessions in one order; set order depends on the process."""
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "b", list(range(50)))
    _finished_turn(manager, holder, requests, "a", list(range(100, 150)))
    order = []
    release_one = holder._release
    holder._release = lambda session_id: (
        order.append(session_id),
        release_one(session_id),
    )

    holder.release(session_ids=["b", "a"])

    assert order == ["a", "b"]


def test_evict_until_free_releases_the_session_idle_longest_first(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "old", list(range(50)))
    _finished_turn(manager, holder, requests, "new", list(range(500, 550)))

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.evict_until_free()
    # Only "old" was released: releasing "new" now frees the rest.
    assert holder.num_sessions == 1 and holder.num_evicted == 1
    assert holder.num_stall_evicted == 0
    holder.release(session_ids=["new"])
    assert holder.num_sessions == 0


def test_evict_until_free_skips_a_session_with_a_running_request(manager):
    requests = {}
    busy = _request("busy", list(range(50)))
    holder = _holder(manager, requests, running=[busy])
    requests["busy"] = busy
    holder.track("busy", session_id="busy", group_id=0)
    _prefill(manager, busy)
    holder.after_step()

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.evict_until_free()
    assert holder.num_sessions == 1 and holder.num_evicted == 0


def test_evict_until_free_releases_a_session_whose_next_request_is_waiting(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    # The next turn is in vLLM but not scheduled yet: only the hold keeps its blocks.
    requests["s/turn2"] = _request("s/turn2", list(range(60)))
    holder.track("s/turn2", session_id="s", group_id=0)

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.evict_until_free()
    assert holder.num_sessions == 0 and holder.num_evicted == 1


def test_release_before_prefill_takes_no_hold(manager):
    requests = {}
    holder = _holder(manager, requests)
    request = _request("t1", list(range(50)))
    requests["t1"] = request
    holder.track("t1", session_id="s", group_id=7)
    holder.release(session_ids=["s"], group_ids=[7])
    _prefill(manager, request)
    holder.after_step()
    assert holder.num_sessions == 0


def test_next_turn_moves_the_hold_and_release_frees_everything(manager):
    requests = {}
    holder = _holder(manager, requests)
    free_before = manager.block_pool.get_num_free_blocks()
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn1)
    holder.after_step()
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    manager.free(turn1)
    del requests["t1"]
    holder.after_step()

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    requests["t2"] = turn2
    holder.track("t2", session_id="s", group_id=0)
    _prefill(manager, turn2)
    holder.after_step()
    manager.free(turn2)
    del requests["t2"]
    holder.after_step()
    assert holder.num_sessions == 1

    holder.release(session_ids=["s"])
    assert manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize(
    "cpu_hit_tokens, expected",
    [(0, (48, 0, 16)), (16, (48, 16, 0))],
)
def test_a_later_turn_counts_its_prefix_as_gpu_hit_cpu_hit_or_recomputed(
    manager, cpu_hit_tokens, expected
):
    """Turn 1 computed 64 tokens in full blocks; the hold keeps its prompt end (48)."""
    requests = {}
    holder = _holder(manager, requests)
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn1)
    holder.after_step()
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    manager.free(turn1)
    del requests["t1"]
    holder.after_step()
    # A session's first turn has nothing to reuse.
    assert holder.gpu_hit_prefix_tokens == holder.recomputed_prefix_tokens == 0

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    requests["t2"] = turn2
    holder.track("t2", session_id="s", group_id=0)
    gpu_hit_tokens = _prefill(manager, turn2)
    # vLLM fills these at the first schedule (`_prefill` stands in for it); a CPU load extends
    # the GPU hit.
    turn2.prefill_stats.set(
        num_prompt_tokens=turn2.num_prompt_tokens,
        num_local_cached_tokens=gpu_hit_tokens,
        num_external_cached_tokens=cpu_hit_tokens,
    )
    holder.after_step()

    assert gpu_hit_tokens == 48
    assert (
        holder.gpu_hit_prefix_tokens,
        holder.cpu_hit_prefix_tokens,
        holder.recomputed_prefix_tokens,
    ) == expected


def test_held_blocks_count_a_block_two_siblings_hold_once(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "a", list(range(50)))
    held_by_a = holder.num_held_blocks
    # The same prompt: "b" hits and holds the blocks "a" holds.
    _finished_turn(manager, holder, requests, "b", list(range(50)))

    assert held_by_a > 0
    assert holder.num_sessions == 2 and holder.num_held_blocks == held_by_a
    holder.release(session_ids=["a"])
    assert holder.num_held_blocks == held_by_a
    holder.release(session_ids=["b"])
    assert holder.num_held_blocks == 0


# --- CPU offload: ended sessions' chunks are evicted first ---

_CHURN = OffloadKey(b"churn")


def _cpu_tier() -> CPUOffloadingManager:
    """vLLM's CPU offload manager with its default LRU; 1,000 chunks is more than any test stores."""
    return CPUOffloadingManager(num_chunks=1000)


def _store(cpu_tier: CPUOffloadingManager, keys) -> None:
    """Store `keys` as vLLM's write-through does: prepare the store, then complete it."""
    keys = list(keys)
    context = ReqContext(req_id="store")
    cpu_tier.prepare_store(keys, context)
    cpu_tier.complete_store(keys, context)


def _offload_keys(request: Request) -> list[OffloadKey]:
    """CPU chunk keys of every full block of `request`, in all 4 KV cache groups."""
    return [make_offload_key(h, g) for h in request.block_hashes for g in range(4)]


def _eviction_order(cpu_tier: CPUOffloadingManager) -> list[OffloadKey]:
    """Stored chunks, evicted first on the left."""
    return list(cast(LRUCachePolicy, cpu_tier._policy).evictable_chunks)


def test_release_evicts_an_ended_session_first_except_the_prefix_a_live_sibling_uses(
    manager,
):
    requests, cpu_tier = {}, _cpu_tier()
    holder = _holder(manager, requests, offload_policy=cpu_tier._policy)
    prompt = list(range(48))  # 3 blocks both siblings share
    s1 = _finished_turn(manager, holder, requests, "s1", prompt + list(range(100, 134)))
    s2 = _finished_turn(manager, holder, requests, "s2", prompt + list(range(200, 234)))
    _store(cpu_tier, [_CHURN])
    _store(cpu_tier, dict.fromkeys(_offload_keys(s1) + _offload_keys(s2)))
    # 3 shared blocks x 4 groups, then each sibling's own 2 blocks x 4 groups.
    shared, s1_own, s2_own = (
        _offload_keys(s1)[:12],
        _offload_keys(s1)[12:],
        _offload_keys(s2)[12:],
    )

    holder.release(session_ids=["s1"])
    order = _eviction_order(cpu_tier)
    assert set(order[: len(s1_own)]) == set(s1_own)
    assert order[len(s1_own)] == _CHURN  # the shared prefix stays behind it

    holder.release(session_ids=["s2"])
    order = _eviction_order(cpu_tier)
    assert set(order[: len(shared) + len(s2_own)]) == set(shared + s2_own)


def test_a_sibling_before_its_first_prefill_keeps_the_shared_prefix_in_place(manager):
    requests, cpu_tier = {}, _cpu_tier()
    holder = _holder(manager, requests, hold=False, offload_policy=cpu_tier._policy)
    prompt = list(range(48))  # 3 blocks both siblings share
    s1 = _finished_turn(manager, holder, requests, "s1", prompt + list(range(100, 134)))
    # s2's first turn is in vLLM but not prefilled yet.
    requests["s2"] = _request("s2", prompt + list(range(200, 234)))
    holder.track("s2", session_id="s2", group_id=0)
    _store(cpu_tier, [_CHURN] + _offload_keys(s1))

    holder.release(session_ids=["s1"])

    order = _eviction_order(cpu_tier)
    assert set(order[:8]) == set(_offload_keys(s1)[12:])  # s1's own 2 blocks x 4 groups
    assert order[8] == _CHURN


def test_completion_blocks_of_an_ended_session_are_evicted_first(manager):
    requests, cpu_tier = {}, _cpu_tier()
    holder = _holder(manager, requests, hold=False, offload_policy=cpu_tier._policy)
    turn = _request("t1", list(range(50)))
    requests["t1"] = turn
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn)
    holder.after_step()
    for t in range(20):  # decode fills a 4th block: 50 prompt + 21 sampled tokens
        _schedule(manager, turn, 1, sample=100 + t)
    manager.free(turn)
    del requests["t1"]
    holder.after_step()
    assert len(turn.block_hashes) == 4
    _store(cpu_tier, [_CHURN] + _offload_keys(turn))

    holder.release(session_ids=["s"])

    assert _eviction_order(cpu_tier)[-1] == _CHURN


def test_offload_without_hold_holds_nothing_and_evicts_a_finished_group_first(manager):
    requests, cpu_tier = {}, _cpu_tier()
    holder = _holder(manager, requests, hold=False, offload_policy=cpu_tier._policy)
    free_before = manager.block_pool.get_num_free_blocks()
    s1 = _finished_turn(manager, holder, requests, "s1", list(range(50)), group_id=5)
    s2 = _finished_turn(manager, holder, requests, "s2", list(range(500, 550)), 5)
    _store(cpu_tier, [_CHURN] + _offload_keys(s1) + _offload_keys(s2))
    assert holder.num_sessions == 0
    assert manager.block_pool.get_num_free_blocks() == free_before

    holder.release(group_ids=[5])

    assert _eviction_order(cpu_tier)[-1] == _CHURN


def test_cpu_used_fraction_counts_stored_chunks(manager):
    """vLLM's usage gauge counts only chunks a transfer pins, so a full tier reads 0 there."""
    cpu_tier = _cpu_tier()
    holder = _holder(manager, {}, hold=False, offload_policy=cpu_tier._policy)
    _store(cpu_tier, [OffloadKey(bytes([i])) for i in range(250)])

    assert holder.cpu_used_fraction() == 0.25
    vllm_gauges = cpu_tier.get_stats().reduce()
    assert vllm_gauges["vllm:kv_offload_cpu_cache_usage_perc"] == 0.0


def test_free_floor_release_keeps_the_cpu_chunks_of_a_live_session(manager):
    """A session dropped from GPU for room is still alive: its CPU copy is what it reloads."""
    requests, cpu_tier = {}, _cpu_tier()
    holder = _holder(manager, requests, offload_policy=cpu_tier._policy)
    old = _finished_turn(manager, holder, requests, "old", list(range(50)))
    _finished_turn(manager, holder, requests, "new", list(range(500, 550)))
    _store(cpu_tier, _offload_keys(old) + [_CHURN])
    order_before = _eviction_order(cpu_tier)

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.evict_until_free()

    assert holder.num_evicted == 1
    assert _eviction_order(cpu_tier) == order_before


def test_offload_needs_vllm_lru_policy(manager):
    with pytest.raises(
        RuntimeError, match="needs vLLM's LRUCachePolicy, got ARCCachePolicy"
    ):
        _holder(manager, {}, offload_policy=ARCCachePolicy(cache_capacity=8))


def test_free_floor_is_the_watermark_plus_one_prefill_chunk(manager):
    manager.watermark_blocks = 8  # vLLM's watermark=0.1 of the 80-block pool
    scheduler = SimpleNamespace(kv_cache_manager=manager, max_num_scheduled_tokens=100)
    # watermark 8 + 4 KV cache groups x cdiv(100, 16) = 7 chunk blocks
    assert SessionKVHolder(scheduler).free_floor_blocks == 8 + 4 * 7


@pytest.mark.parametrize("queue", ["waiting", "skipped_waiting"])
def test_a_waiting_turn_gets_room_when_held_sessions_fill_the_pool(manager, queue):
    """Nothing runs, free >= floor, and the waiting turn needs more blocks than are free.
    vLLM serves `skipped_waiting` first and leaves a refused request there."""
    requests, waiting = {}, []
    holder = _holder(manager, requests, floor=4, **{queue: waiting})
    for i in range(12):  # 6 held blocks each: 7 of 80 stay free
        _finished_turn(
            manager, holder, requests, f"s{i}", [1000 * i + j for j in range(50)]
        )
    # 10 attention blocks + GDN state: more than the 7 free.
    head = _request("head", list(range(50_000, 50_160)))
    requests["head"] = head
    waiting.append(head)

    # The engine loop with an empty batch: the holder, then vLLM's admission check.
    for _ in range(20):
        holder.evict_until_free()
        manager.new_step_starts()
        if manager.allocate_slots(
            head, head.num_tokens, full_sequence_must_fit=True, has_scheduled_reqs=False
        ):
            break
        holder.after_step()
    else:
        pytest.fail("held sessions stalled the engine: the waiting turn never fit")
    assert holder.num_stall_evicted == holder.num_evicted > 0


def test_a_turn_loading_kv_from_cpu_releases_nothing(manager):
    """A request whose CPU load is in flight waits for the load, not for blocks."""
    loading = _request("loading", list(range(60)))
    loading.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    requests = {}
    holder = _holder(manager, requests, skipped_waiting=[loading])
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    assert holder.num_sessions == 1
    assert holder.num_evicted == 0
