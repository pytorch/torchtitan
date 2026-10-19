# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VLLMSessionKVHolder on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align"
groups, 16-token blocks standing in for Qwen3.5's 1,152."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
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
from vllm.v1.request import Request

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


def _scheduler(manager, max_num_scheduled_tokens=0, **attrs) -> SimpleNamespace:
    """The vLLM scheduler attributes the holder reads."""
    return SimpleNamespace(
        kv_cache_manager=manager,
        vllm_config=SimpleNamespace(speculative_config=None),
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        **attrs,
    )


def _holder(
    manager, requests, floor=0, running=(), waiting=(), skipped_waiting=()
) -> VLLMSessionKVHolder:
    scheduler = _scheduler(
        manager,
        requests=requests,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped_waiting,
    )
    holder = VLLMSessionKVHolder.Config().build(scheduler=scheduler)
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


def _finished_turn(manager, holder, requests, session: str, tokens: list[int]) -> None:
    """One turn of `session`: admit, prefill (held), finish."""
    request = _request(session, tokens)
    requests[session] = request
    holder.track(session, session_id=session, group_id=0)
    _prefill(manager, request)
    holder.after_step()
    manager.free(request)
    del requests[session]
    holder.after_step()


def test_a_later_turn_counts_its_prefix_as_gpu_hit_or_recomputed(manager):
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
    # vLLM fills these at the first schedule; `_prefill` stands in for it.
    turn2.prefill_stats.set(
        num_prompt_tokens=turn2.num_prompt_tokens,
        num_local_cached_tokens=gpu_hit_tokens,
        num_external_cached_tokens=0,
    )
    holder.after_step()

    assert gpu_hit_tokens == 48
    # 64 reusable tokens (4 full blocks), 48 hit: the decode block was recomputed.
    assert (holder.gpu_hit_prefix_tokens, holder.recomputed_prefix_tokens) == (48, 16)


def test_group_release_drops_hashes_of_evicted_sessions(manager):
    """An evicted session leaves `_group_sessions`; its group's release must still drop its hashes."""
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    holder.free_floor_blocks = 10**6
    holder._evict_until_free()
    assert "s" not in holder._held and holder.num_evicted == 1
    holder.release(group_ids=[0])
    assert holder._session_hashes == {}


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
    holder.after_step()
    # Only "old" was released: releasing "new" now frees the rest.
    assert holder.num_sessions == 1 and holder.num_evicted == 1
    holder.release(session_ids=["new"])
    assert holder.num_sessions == 0


def test_evict_until_free_releases_a_session_with_a_running_request_last(manager):
    requests = {}
    busy = _request("busy", list(range(50)))
    holder = _holder(manager, requests, running=[busy])
    requests["busy"] = busy
    holder.track("busy", session_id="busy", group_id=0)
    _prefill(manager, busy)
    holder.after_step()
    # One decode step: the prompt-end GDN states leave the request; only the hold keeps them.
    _schedule(manager, busy, 1, sample=8)
    _finished_turn(manager, holder, requests, "idle", list(range(500, 550)))

    # "busy" is held longer, but its request runs: "idle" goes first.
    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    assert list(holder._held) == ["busy"] and holder.num_evicted == 1
    # Still short: the running session's hold goes too, freeing the 3 GDN states only it kept.
    free_before = manager.block_pool.get_num_free_blocks()
    holder.free_floor_blocks = free_before + 1
    holder.after_step()
    assert holder.num_sessions == 0 and holder.num_evicted == 2
    assert manager.block_pool.get_num_free_blocks() == free_before + 3
    # The request kept its own references: finishing it returns the whole pool.
    manager.free(busy)
    assert (
        manager.block_pool.get_num_free_blocks()
        == manager.block_pool.num_gpu_blocks - 1
    )


def test_evict_until_free_releases_a_session_whose_next_request_is_waiting(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    # The next turn is in vLLM but not scheduled yet: only the hold keeps its blocks.
    requests["s/turn2"] = _request("s/turn2", list(range(60)))
    holder.track("s/turn2", session_id="s", group_id=0)

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
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


def test_free_floor_is_the_watermark_plus_one_prefill_chunk(manager):
    manager.watermark_blocks = 8  # vLLM's watermark=0.1 of the 80-block pool
    scheduler = _scheduler(manager, max_num_scheduled_tokens=100)
    holder = VLLMSessionKVHolder.Config().build(scheduler=scheduler)
    # watermark 8 + 4 KV cache groups x cdiv(100, 16) = 7 chunk blocks
    assert holder.free_floor_blocks == 8 + 4 * 7


def test_holder_rejects_partial_block_hits_and_speculative_decoding(manager):
    manager.coordinator.enable_partial_hash_hits = True
    with pytest.raises(ValueError, match="does not support partial-block prefix hits"):
        VLLMSessionKVHolder.Config().build(scheduler=_scheduler(manager))
    manager.coordinator.enable_partial_hash_hits = False
    scheduler = _scheduler(manager)
    scheduler.vllm_config.speculative_config = object()
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        VLLMSessionKVHolder.Config().build(scheduler=scheduler)


@pytest.mark.parametrize("queue", ["waiting", "skipped_waiting"])
def test_a_waiting_turn_gets_room_when_held_sessions_fill_the_pool(manager, queue):
    """Nothing runs, free >= floor, and the waiting turn needs more blocks than are free.
    vLLM serves `skipped_waiting` first and leaves a refused request there."""
    requests, waiting = {}, []
    holder = _holder(manager, requests, **{queue: waiting})
    for i in range(12):  # 6 held blocks each: 7 of 80 stay free
        _finished_turn(
            manager, holder, requests, f"s{i}", [1000 * i + j for j in range(50)]
        )
    # Set after the turns: every after_step checks the floor, which would release during setup.
    holder.free_floor_blocks = 4
    assert holder.num_sessions == 12 and manager.block_pool.get_num_free_blocks() == 7
    # 10 attention blocks + GDN state: more than the 7 free.
    head = _request("head", list(range(50_000, 50_160)))
    requests["head"] = head
    waiting.append(head)

    # The engine loop with an empty batch: vLLM's admission check, then the holder.
    for _ in range(20):
        manager.new_step_starts()
        if manager.allocate_slots(
            head, head.num_tokens, full_sequence_must_fit=True, has_scheduled_reqs=False
        ):
            break
        holder.after_step()
    else:
        pytest.fail("held sessions stalled the engine: the waiting turn never fit")


def test_engine_session_kv_counts_are_deltas_since_the_last_call():
    """Counts (e.g. evictions) cover the time since the previous call; levels are read as is."""
    from types import SimpleNamespace

    from torchtitan.rl.generator.vllm.engine import VLLMEngine

    holder = SimpleNamespace(
        num_sessions=80,
        num_evicted=3,
        num_stall_evicted=0,
        gpu_hit_prefix_tokens=0,
        recomputed_prefix_tokens=0,
    )
    engine = object.__new__(VLLMEngine)
    engine._session_kv_holder = holder
    engine._session_kv_totals = {}
    scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(usage=0.5))
    engine._engine = SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler))
    )

    def evicted(metrics):
        [metric] = [x for x in metrics if x.key == "generator/session_kv/evicted_sessions"]
        return metric.value.value

    assert evicted(engine.session_kv_metrics()) == 3
    holder.num_evicted = 5
    assert evicted(engine.session_kv_metrics()) == 2
