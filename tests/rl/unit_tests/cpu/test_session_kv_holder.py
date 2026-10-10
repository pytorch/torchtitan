# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SessionKVHolder on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align" groups,
16-token blocks standing in for Qwen3.5's 1,152 (and an attention-only manager where block counts
matter)."""

from types import SimpleNamespace

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


@pytest.fixture
def attention_manager() -> KVCacheManager:
    """One attention group of 100 usable blocks (plus vLLM's null block)."""
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=101,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention)],
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


def _holder(manager, requests, running=()) -> SessionKVHolder:
    scheduler = SimpleNamespace(
        kv_cache_manager=manager, requests=requests, running=running, connector=None
    )
    return SessionKVHolder(scheduler)


def _tiers(manager) -> tuple[list, list]:
    """The free blocks of the ordinary and the live tier, in eviction order."""
    queue = manager.block_pool.free_block_queue
    return queue._ordinary.get_all_free_blocks(), queue._live.get_all_free_blocks()


def _check_queue(manager) -> None:
    """Every free block sits in exactly one tier: the live tier iff a live session marks it."""
    queue = manager.block_pool.free_block_queue
    ordinary, live = _tiers(manager)
    assert len(ordinary) == queue._ordinary.num_free_blocks
    assert len(live) == queue._live.num_free_blocks
    assert all(block.block_id in queue._marks for block in live)
    assert not any(block.block_id in queue._marks for block in ordinary)
    free = [
        b.block_id
        for b in manager.block_pool.blocks
        if b.ref_cnt == 0 and not b.is_null
    ]
    assert sorted(b.block_id for b in ordinary + live) == sorted(free)


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
    _check_queue(manager)
    assert holder.num_reused_blocks == 0

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


def test_held_blocks_stay_free_and_release_moves_them_to_the_evict_first_end(
    manager,
):
    requests = {}
    holder = _holder(manager, requests)
    free_before = manager.block_pool.get_num_free_blocks()
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=3)
    _prefill(manager, turn1)
    holder.after_step()
    manager.free(turn1)
    del requests["t1"]
    holder.after_step()
    held = [block for block, _ in holder._session_marks["s"]]
    assert held and manager.block_pool.get_num_free_blocks() == free_before
    # In the order vLLM freed them (tail first), not the order they were marked in.
    assert {b.block_id for b in _tiers(manager)[1]} == {b.block_id for b in held}
    _check_queue(manager)

    holder.release(group_ids=[3])
    assert holder.num_sessions == 0
    ordinary, live = _tiers(manager)
    assert not live and ordinary[: len(held)] == held[::-1]
    assert manager.block_pool.get_num_free_blocks() == free_before
    _check_queue(manager)


def test_release_unmarks_sessions_in_sorted_order(manager):
    """Every TP rank unmarks a release's sessions in one order; set order depends on the process."""
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "b", list(range(50)))
    _finished_turn(manager, holder, requests, "a", list(range(100, 150)))
    order = []
    unmark_one = holder._unmark
    holder._unmark = lambda session_id: (
        order.append(session_id),
        unmark_one(session_id),
    )

    holder.release(session_ids=["b", "a"])

    assert order == ["a", "b"]


def test_a_request_larger_than_the_unheld_blocks_is_admitted(attention_manager):
    """Held blocks count as free: vLLM admits a request that needs them, taking the oldest."""
    manager = attention_manager
    requests = {}
    holder = _holder(manager, requests)
    # 8 sessions finish a turn of 10 full blocks (+1 token): 80 blocks held, 20 not.
    for s in range(8):
        tokens = [10_000 * (s + 1) + j for j in range(10 * BLOCK + 1)]
        _finished_turn(manager, holder, requests, f"s{s}", tokens)
    ordinary, live = _tiers(manager)
    assert (len(ordinary), len(live)) == (20, 80)

    # Nothing runs, and a fresh request needs 25 blocks.
    big = _request("big", [7 + j for j in range(25 * BLOCK)])
    manager.new_step_starts()
    assert manager.allocate_slots(
        big, big.num_tokens, full_sequence_must_fit=True, has_scheduled_reqs=False
    )
    # The 5 blocks beyond the unheld ones come from the session held longest.
    assert holder.num_reused_blocks == 5
    _check_queue(manager)

    # Releasing skips the marks on the 5 reused blocks, which now hold `big`'s KV.
    holder.release(session_ids=[f"s{s}" for s in range(8)])
    assert holder.num_sessions == 0 and not manager.block_pool.free_block_queue._marks
    _check_queue(manager)
    manager.free(big)
    _check_queue(manager)


def test_shared_prefix_stays_held_until_its_last_session_ends(attention_manager):
    manager = attention_manager
    requests = {}
    holder = _holder(manager, requests)
    prefix = list(range(10 * BLOCK))
    _finished_turn(manager, holder, requests, "a", prefix + [1, 2, 3])
    _finished_turn(manager, holder, requests, "b", prefix + [4, 5, 6])
    shared = {block.block_id for block, _ in holder._session_marks["a"]}
    assert shared == {block.block_id for block, _ in holder._session_marks["b"]}

    holder.release(session_ids=["a"])
    assert {block.block_id for block in _tiers(manager)[1]} == shared
    holder.release(session_ids=["b"])
    assert not _tiers(manager)[1]
    _check_queue(manager)


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


def test_next_turn_moves_the_hold_and_release_unmarks_everything(manager):
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

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    requests["t2"] = turn2
    holder.track("t2", session_id="s", group_id=0)
    _prefill(manager, turn2)
    holder.after_step()
    manager.free(turn2)
    del requests["t2"]
    holder.after_step()
    assert holder.num_sessions == 1
    _check_queue(manager)

    holder.release(session_ids=["s"])
    assert not _tiers(manager)[1] and not manager.block_pool.free_block_queue._marks
    _check_queue(manager)
