# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VLLMSessionKVEvictor on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align"
groups with dense retention (as `VLLMEngine` sets it), 16-token blocks standing in for
Qwen3.5's 1,152."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.rl.generator.engine import KVCacheCapacity
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.generator.vllm.session_kv_evictor import VLLMSessionKVEvictor
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

BLOCK = 16
ATTENTION_KV_CACHE_GROUP_ID = 0
MAMBA_KV_CACHE_GROUP_IDS = (1, 2, 3)


@pytest.fixture
def manager(request) -> KVCacheManager:
    """Indirect param: the number of Mamba groups (default 3; 0 is a dense model)."""
    return _hybrid_manager(
        getattr(request, "param", len(MAMBA_KV_CACHE_GROUP_IDS)), use_eagle=False
    )


def _hybrid_manager(num_mamba_groups: int, *, use_eagle: bool) -> KVCacheManager:
    """1 attention group and `num_mamba_groups` align-mode Mamba groups, 16-token blocks;
    `use_eagle` drops the last block of each attention hit, as vLLM does with MTP drafts."""
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    groups = [KVCacheGroupSpec(["attn"], attention)] + [
        KVCacheGroupSpec([f"mamba{i}"], mamba) for i in range(num_mamba_groups)
    ]
    config = KVCacheConfig(
        num_blocks=120,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=None,
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
        use_eagle=use_eagle,
    )


@pytest.fixture
def scheduler(manager) -> SimpleNamespace:
    """The vLLM scheduler attributes the evictor reads."""
    return SimpleNamespace(kv_cache_manager=manager, requests={})


@pytest.fixture
def evictor(scheduler) -> VLLMSessionKVEvictor:
    return VLLMSessionKVEvictor(scheduler=scheduler)


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


def _prefill(manager, request, sample: int = 7, *, use_eagle: bool = False) -> int:
    """Block-aligned chunk, then the tail (like vLLM's Mamba-aligned split, one block earlier
    with MTP drafts); returns the hit."""
    manager.new_step_starts()
    blocks, hit, _ = manager.get_computed_blocks(request)
    num_prompt = request.num_prompt_tokens
    aligned = num_prompt - num_prompt % BLOCK
    if use_eagle:
        aligned = max(aligned - BLOCK, 0)
    first_hit = hit
    if aligned > hit:
        _schedule(manager, request, aligned - hit, blocks, hit)
        blocks, hit = None, 0
    _schedule(
        manager,
        request,
        num_prompt - request.num_computed_tokens - hit,
        blocks,
        hit,
        sample,
    )
    return first_hit


def _admit(scheduler, evictor, request_id: str, session_id: str, group_id: int, tokens):
    """Add a request to vLLM and to the evictor; the fake keeps vLLM's id equal to ours."""
    request = _request(request_id, tokens)
    scheduler.requests[request_id] = request
    evictor.track(request_id, session_id=session_id, group_id=group_id)
    return request


def _finish(manager, scheduler, evictor, request) -> None:
    manager.free(request)
    del scheduler.requests[request.request_id]
    evictor.after_step()


def _turn(
    manager,
    scheduler,
    evictor,
    request_id: str,
    *,
    session_id: str,
    group_id: int,
    tokens: list[int],
    output_tokens: list[int],
) -> Request:
    """One finished turn: admit, prefill, decode ``output_tokens``, finish."""
    request = _admit(scheduler, evictor, request_id, session_id, group_id, tokens)
    _prefill(manager, request, sample=output_tokens[0])
    for token in output_tokens[1:]:
        _schedule(manager, request, 1, sample=token)
    _finish(manager, scheduler, evictor, request)
    return request


def _cached_indices(manager, request, kv_cache_group_id: int) -> list[int]:
    """Indices of ``request``'s full blocks still cached in KV cache group ``kv_cache_group_id``."""
    get = manager.block_pool.cached_block_hash_to_block.get_one_block
    return [
        index
        for index, block_hash in enumerate(request.block_hashes)
        if get(make_block_hash_with_group_id(block_hash, kv_cache_group_id)) is not None
    ]


def _num_cached_blocks(manager, request) -> int:
    return sum(
        len(_cached_indices(manager, request, kv_cache_group_id))
        for kv_cache_group_id in (
            ATTENTION_KV_CACHE_GROUP_ID,
            *MAMBA_KV_CACHE_GROUP_IDS,
        )
    )


def _num_copies(manager, key) -> int:
    """Blocks cached under ``key``, a block hash with its KV cache group id."""
    return sum(block.block_hash == key for block in manager.block_pool.blocks)


def _num_free_uncached_blocks(manager) -> int:
    queue = manager.block_pool.free_block_queue
    return sum(block.block_hash is None for block in queue.iter_blocks_after(None))


def test_an_ended_rollouts_blocks_are_reused_before_a_live_rollouts(
    manager, scheduler, evictor
):
    # "live" finished first, so vLLM's LRU alone would hand out its blocks before "ended"'s.
    live = _turn(
        manager,
        scheduler,
        evictor,
        "live/turn=0",
        session_id="live",
        group_id=0,
        tokens=list(range(50)),
        output_tokens=list(range(100, 120)),
    )
    ended_turn0 = _turn(
        manager,
        scheduler,
        evictor,
        "ended/turn=0",
        session_id="ended",
        group_id=1,
        tokens=list(range(500, 550)),
        output_tokens=list(range(600, 620)),
    )
    # Its second turn repeats turn 0's history, which must still count one owner.
    ended = _turn(
        manager,
        scheduler,
        evictor,
        "ended/turn=1",
        session_id="ended",
        group_id=1,
        tokens=list(ended_turn0.all_token_ids) + list(range(700, 730)),
        output_tokens=list(range(800, 820)),
    )
    num_ended_blocks = _num_cached_blocks(manager, ended)
    num_free_uncached = _num_free_uncached_blocks(manager)

    # The rollout ended, and with it its group of one.
    evictor.release(session_ids=["ended"], group_ids=[1])

    assert _num_cached_blocks(manager, ended) == 0
    assert evictor.num_evicted_blocks == num_ended_blocks
    # They head the free queue in block id order, so every TP rank leaves the same queue.
    queue = manager.block_pool.free_block_queue.iter_blocks_after(None)
    front = [next(queue).block_id for _ in range(num_ended_blocks)]
    assert front == sorted(front)
    # Other requests take as many blocks as were free and uncached, plus the ended rollout's.
    manager.block_pool.get_new_blocks(num_free_uncached + num_ended_blocks)
    # The live rollout's next turn still resumes after its 4 full blocks (64 of 73 tokens).
    next_turn = _request("live/turn=1", list(live.all_token_ids) + [5, 6, 7])
    _, num_hit_tokens, _ = manager.get_computed_blocks(next_turn)
    assert num_hit_tokens == 64
    # Once every rollout ended, no hash keeps an owner.
    evictor.release(session_ids=["live"], group_ids=[0])
    assert evictor._num_requests_by_hash == {}


def test_an_ended_groups_rollouts_are_evicted_but_blocks_a_live_group_shares_stay(
    manager, scheduler, evictor
):
    # Two groups with one policy version (cache salt) share a 32-token system prompt.
    system_prompt = list(range(32))
    ended = _turn(
        manager,
        scheduler,
        evictor,
        "group=1/rollout=0/turn=0",
        session_id="group=1/rollout=0",
        group_id=1,
        tokens=system_prompt + list(range(500, 518)),
        output_tokens=list(range(600, 620)),
    )
    live = _turn(
        manager,
        scheduler,
        evictor,
        "group=2/rollout=0/turn=0",
        session_id="group=2/rollout=0",
        group_id=2,
        tokens=system_prompt + list(range(700, 718)),
        output_tokens=list(range(800, 820)),
    )

    # Group 1 ends; its rollout never released its own session.
    evictor.release(session_ids=[], group_ids=[1])

    # Only the 2 system prompt blocks of group 1's rollout stay cached.
    assert _cached_indices(manager, ended, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1]
    assert evictor.num_sessions == 1
    # Group 1 is forgotten; group 2's rollout is still live.
    assert evictor._group_sessions == {2: {"group=2/rollout=0"}}
    # Group 2's next turn still resumes after its 4 full blocks (64 of 73 tokens).
    next_turn = _request(
        "group=2/rollout=0/turn=1", list(live.all_token_ids) + [5, 6, 7]
    )
    _, num_hit_tokens, _ = manager.get_computed_blocks(next_turn)
    assert num_hit_tokens == 64


def test_a_rollout_released_while_its_request_runs_is_evicted_when_it_finishes(
    manager, scheduler, evictor
):
    request = _admit(scheduler, evictor, "s/turn=0", "s", 0, list(range(50)))
    _prefill(manager, request)

    evictor.release(session_ids=["s"], group_ids=[])
    for token in range(100, 120):
        _schedule(manager, request, 1, sample=token)
    _finish(manager, scheduler, evictor, request)

    assert _num_cached_blocks(manager, request) == 0
    assert evictor.num_sessions == 0


def test_a_late_request_of_a_released_group_starts_no_session(
    manager, scheduler, evictor
):
    # The group ends while an abandoned attempt is still on its way; the attempt arrives after.
    evictor.release(session_ids=[], group_ids=[0])
    request = _admit(scheduler, evictor, "s/turn=1", "s", 0, list(range(50)))
    _prefill(manager, request)
    _finish(manager, scheduler, evictor, request)

    assert evictor.num_sessions == 0
    # Left to vLLM's LRU: still cached.
    assert _cached_indices(manager, request, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2]


@pytest.mark.parametrize("manager", [0, 3], ids=["dense", "hybrid"], indirect=True)
def test_a_preempted_requests_output_stays_cached_when_a_group_sharing_it_ends(
    manager, scheduler, evictor
):
    # Rollout "a" decodes 40 tokens after its 50-token prompt. Then vLLM preempts it: its blocks
    # are freed (they stay cached) and it waits to resume.
    a = _admit(scheduler, evictor, "a/turn=0", "a", 1, list(range(50)))
    _prefill(manager, a, sample=100)
    for token in range(101, 140):
        _schedule(manager, a, 1, sample=token)
    manager.free(a)
    a.num_computed_tokens = 0
    # A judge in another group reads a's history (same cache salt), and its group ends.
    _turn(
        manager,
        scheduler,
        evictor,
        "judge/turn=0",
        session_id="judge",
        group_id=2,
        tokens=list(a.all_token_ids) + [7, 8, 9],
        output_tokens=[1],
    )
    evictor.release(session_ids=[], group_ids=[2])

    # a still resumes after all 5 of its full blocks (80 of its 90 tokens).
    _, num_hit_tokens, _ = manager.get_computed_blocks(a)
    assert num_hit_tokens == 80


def test_a_retry_admitted_while_a_request_of_its_session_is_preempted_keeps_its_history(
    manager, scheduler, evictor
):
    turn0 = _turn(
        manager,
        scheduler,
        evictor,
        "s/turn=0",
        session_id="s",
        group_id=0,
        tokens=list(range(50)),
        output_tokens=list(range(100, 140)),
    )
    # Turn 1 decodes 16 tokens, then vLLM preempts it: its blocks are freed and stay cached.
    turn1 = _admit(
        scheduler, evictor, "s/turn=1", "s", 0, list(turn0.all_token_ids) + [7] * 10
    )
    _prefill(manager, turn1, sample=400)
    for token in range(401, 416):
        _schedule(manager, turn1, 1, sample=token)
    manager.free(turn1)
    turn1.num_computed_tokens = 0
    # A retry with an edited history (a client timeout) arrives while turn 1 is still in vLLM.
    _admit(
        scheduler, evictor, "s/retry", "s", 0, list(range(50)) + list(range(700, 731))
    )

    # Turn 1 still resumes after all 7 of its full blocks (112 of its 116 tokens).
    _, num_hit_tokens, _ = manager.get_computed_blocks(turn1)
    assert num_hit_tokens == 112


def test_a_retry_of_a_running_request_leaves_the_requests_reply_to_evict(
    manager, scheduler, evictor
):
    first = _admit(scheduler, evictor, "s/turn=1", "s", 0, list(range(50)))
    _prefill(manager, first, sample=100)
    # A client-timeout retry with the same prompt arrives while the first attempt still decodes.
    _admit(scheduler, evictor, "s/retry", "s", 0, list(range(50)))
    for token in range(101, 140):
        _schedule(manager, first, 1, sample=token)
    _finish(manager, scheduler, evictor, first)

    # The retry counts the prompt's blocks 0-2; the attempt's reply blocks 3-4 go.
    assert _cached_indices(manager, first, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2]


def test_an_edited_history_releases_the_dropped_reply(manager, scheduler, evictor):
    # Turn 0: a 50-token prompt and a 40-token reply. Full blocks 3 and 4 hold the reply.
    turn0 = _turn(
        manager,
        scheduler,
        evictor,
        "s/turn=0",
        session_id="s",
        group_id=0,
        tokens=list(range(50)),
        output_tokens=list(range(100, 140)),
    )
    # A sibling with the same prompt and its own reply.
    sibling = _turn(
        manager,
        scheduler,
        evictor,
        "o/turn=0",
        session_id="o",
        group_id=0,
        tokens=list(range(50)),
        output_tokens=[1],
    )
    # Turn 1 drops the reply (a chat template that strips old reasoning) and adds 31 tokens.
    _admit(
        scheduler, evictor, "s/turn=1", "s", 0, list(range(50)) + list(range(200, 231))
    )

    # Only the 3 prompt blocks stay cached.
    assert _cached_indices(manager, turn0, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2]
    # s ends: its release must not count the dropped reply again, and the sibling keeps the prompt.
    evictor.release(session_ids=["s"], group_ids=[])
    assert _cached_indices(manager, sibling, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2]


def test_a_duplicate_copy_goes_when_its_request_finishes_and_the_last_once_released(
    manager, scheduler, evictor
):
    prompt = list(range(40))  # 2 full blocks
    running = _admit(scheduler, evictor, "b/turn=0", "b", 2, prompt)
    _prefill(manager, running)
    # Scheduled without a prefix lookup, as in one step with b: a caches a second copy.
    copy = _admit(scheduler, evictor, "a/turn=0", "a", 1, prompt)
    _schedule(manager, copy, len(prompt), sample=7)
    key = make_block_hash_with_group_id(
        running.block_hashes[0], ATTENTION_KV_CACHE_GROUP_ID
    )
    assert _num_copies(manager, key) == 2

    # a's copy goes when a's request finishes: b's request uses the other one.
    _finish(manager, scheduler, evictor, copy)
    assert _num_copies(manager, key) == 1
    evictor.release(session_ids=["a", "b"], group_ids=[])
    # b's request still runs and counts the shared prompt, so its copy stays for now.
    assert _num_copies(manager, key) == 1
    for token in range(100, 120):
        _schedule(manager, running, 1, sample=token)
    _finish(manager, scheduler, evictor, running)
    assert _num_copies(manager, key) == 0
    assert all(block.block_hash is None for block in manager.block_pool.blocks)
    assert evictor._num_requests_by_hash == {} and evictor._ended_requests == {}


@pytest.mark.parametrize("manager", [0, 3], ids=["dense", "hybrid"], indirect=True)
def test_siblings_prefilled_in_one_step_keep_one_copy_of_their_shared_prompt(
    manager, scheduler, evictor
):
    # vLLM schedules a group's first calls in one step, before either's prompt is cached:
    # each sibling computes and caches its own copy of the 40-token prompt (2 full blocks).
    prompt = list(range(40))
    siblings = [
        _admit(scheduler, evictor, f"g/{i}/turn=0", f"g/{i}", 0, prompt)
        for i in range(2)
    ]
    for request in siblings:
        _schedule(manager, request, 32)
    for i, request in enumerate(siblings):
        _schedule(manager, request, 8, sample=100 * (i + 1))
    cached_keys = [
        key
        for block_hash in siblings[0].block_hashes[:2]
        for kv_cache_group_id in range(manager.num_kv_cache_groups)
        if _num_copies(
            manager,
            key := make_block_hash_with_group_id(block_hash, kv_cache_group_id),
        )
    ]
    assert [_num_copies(manager, key) for key in cached_keys] == [2] * len(cached_keys)

    # Sibling 0 ends its turn while sibling 1 still decodes; replies of 20 and 30 tokens.
    for token in range(101, 120):
        _schedule(manager, siblings[0], 1, sample=token)
    _finish(manager, scheduler, evictor, siblings[0])
    for token in range(201, 230):
        _schedule(manager, siblings[1], 1, sample=token)
    _finish(manager, scheduler, evictor, siblings[1])

    # One copy of each shared block stays, and both next turns resume after their full blocks.
    assert [_num_copies(manager, key) for key in cached_keys] == [1] * len(cached_keys)
    hits = []
    for i, request in enumerate(siblings):
        next_turn = _request(f"g/{i}/turn=1", list(request.all_token_ids) + [5, 6, 7])
        _, num_hit_tokens, _ = manager.get_computed_blocks(next_turn)
        hits.append(num_hit_tokens)
    assert hits == [48, 64]


@pytest.mark.parametrize("manager", [0, 3], ids=["dense", "hybrid"], indirect=True)
def test_a_copy_in_use_when_its_sibling_leaves_is_the_copy_that_stays(
    manager, scheduler, evictor
):
    # Siblings a and b prefill one 40-token prompt in one step: 2 copies of its 2 full blocks.
    prompt = list(range(40))
    a, b = (
        _admit(scheduler, evictor, f"g/{i}/turn=0", f"g/{i}", 0, prompt)
        for i in range(2)
    )
    for request in (a, b):
        _schedule(manager, request, 32)
    for request, sample in ((a, 100), (b, 200)):
        _schedule(manager, request, 8, sample=sample)
    cached_keys = {
        key
        for block_hash in a.block_hashes[:2]
        for kv_cache_group_id in range(manager.num_kv_cache_groups)
        if _num_copies(
            manager,
            key := make_block_hash_with_group_id(block_hash, kv_cache_group_id),
        )
    }

    # a leaves while b decodes on its own copy: a's free copy goes, though lookups return it first.
    for token in range(101, 110):
        _schedule(manager, a, 1, sample=token)
    _finish(manager, scheduler, evictor, a)
    assert {_num_copies(manager, key) for key in cached_keys} == {1}

    # b leaves while a KV connector's save still holds its copy; the save then frees it.
    for token in range(201, 210):
        _schedule(manager, b, 1, sample=token)
    held = [
        block for block in manager.block_pool.blocks if block.block_hash in cached_keys
    ]
    manager.block_pool.touch(held)
    _finish(manager, scheduler, evictor, b)
    manager.block_pool.free_blocks(held)
    evictor.after_step()
    assert {_num_copies(manager, key) for key in cached_keys} == {1}


def test_a_kv_cache_group_with_blocks_larger_than_the_hash_block_is_evicted_too():
    # Sliding-window layers with 1 KV head and full layers with 2 get 32- and 16-token blocks
    # (KV cache groups 0 and 1); vLLM hashes every 16 tokens.
    init_none_hash(sha256)
    sliding_window = SlidingWindowSpec(
        block_size=2 * BLOCK,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=4096,
    )
    full = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=120,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["sliding_window"], sliding_window),
            KVCacheGroupSpec(["full"], full),
        ],
        prefix_cache_retention_interval=None,
    )
    manager = KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=2 * BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(kv_cache_manager=manager, requests={})
    evictor = VLLMSessionKVEvictor(scheduler=scheduler)

    def run(request_id: str, session_id: str, group_id: int, tokens) -> Request:
        request = _admit(scheduler, evictor, request_id, session_id, group_id, tokens)
        blocks, hit, _ = manager.get_computed_blocks(request)
        _schedule(manager, request, len(tokens) - hit, blocks, hit, sample=7)
        _finish(manager, scheduler, evictor, request)
        return request

    shared = list(range(64))
    a = run("a/turn=0", "a", 1, shared + list(range(100, 164)))
    b = run("b/turn=0", "b", 2, shared + list(range(200, 264)))
    # vLLM caches each 32-token block under its second 16-token hash.
    assert _cached_indices(manager, a, 0) == [1, 3, 5, 7]

    evictor.release(session_ids=[], group_ids=[1])

    # a's own blocks leave both groups; the prompt b shares stays.
    assert _cached_indices(manager, a, 0) == [1, 3]
    assert _cached_indices(manager, a, 1) == [0, 1, 2, 3]
    assert _cached_indices(manager, b, 0) == [1, 3, 5, 7]
    assert _cached_indices(manager, b, 1) == list(range(8))


@pytest.mark.parametrize("counted_again", [False, True])
def test_a_copy_a_kv_connector_holds_keeps_its_hash_until_the_connector_frees_it(
    manager, scheduler, evictor, counted_again
):
    # Rollouts a and b prefilled one prompt in the same step, so vLLM cached 2 copies of it.
    # vLLM's CPU offload holds a block while it copies it, and asserts the block still has
    # its hash when the copy completes.
    prompt = list(range(40))  # 2 full blocks
    a = _admit(scheduler, evictor, "a/turn=0", "a", 1, prompt)
    b = _admit(scheduler, evictor, "b/turn=0", "b", 2, prompt)
    _schedule(manager, a, len(prompt), sample=7)
    _schedule(manager, b, len(prompt), sample=7)
    key = make_block_hash_with_group_id(a.block_hashes[0], ATTENTION_KV_CACHE_GROUP_ID)
    get = manager.block_pool.cached_block_hash_to_block.get_one_block
    held = get(key)  # the copy vLLM's lookup returns first
    [free] = [
        block
        for block in manager.block_pool.blocks
        if block.block_hash == key and block is not held
    ]
    manager.block_pool.touch([held])
    _finish(manager, scheduler, evictor, a)
    _finish(manager, scheduler, evictor, b)

    evictor.release(session_ids=["a", "b"], group_ids=[])

    assert held.block_hash == key and held.ref_cnt == 1
    assert free.block_hash is None
    assert get(key) is held

    if counted_again:
        # A new rollout's prompt counts the hash before the copy completes.
        _admit(scheduler, evictor, "c/turn=0", "c", 3, prompt)
    manager.block_pool.free_blocks([held])
    evictor.after_step()

    if counted_again:
        assert held.block_hash == key
    else:
        free_block_queue = manager.block_pool.free_block_queue
        assert held.block_hash is None and get(key) is None
        assert next(free_block_queue.iter_blocks_after(None)) is held
    assert evictor._held_block_hashes == set()


def test_cpu_release_hints_wait_for_decode_and_saves_and_skip_a_shared_prefix(
    manager, scheduler
):
    connector = Mock()
    connector.get_pending_save_ids.return_value = {0}
    evictor = VLLMSessionKVEvictor(scheduler=scheduler, offload_connector=connector)
    first = _admit(scheduler, evictor, "first", "session-0", 0, list(range(50)))
    _prefill(manager, first)
    evictor.after_step()
    shared_hashes = set(first.block_hashes)
    evictor.release(session_ids=["session-0"], group_ids=[])
    connector.release_session_cache.assert_not_called()

    # The released request decodes one more block, then leaves vLLM while save 0 runs.
    for token in range(BLOCK):
        _schedule(manager, first, 1, sample=100 + token)
    decode_hashes = set(first.block_hashes) - shared_hashes
    assert decode_hashes
    _finish(manager, scheduler, evictor, first)
    assert evictor.has_pending_work
    connector.release_session_cache.assert_not_called()

    # Another session reuses the prefix before the hint; a newer save doesn't delay it.
    second = _admit(scheduler, evictor, "second", "session-1", 1, list(range(50)))
    _prefill(manager, second)
    connector.get_pending_save_ids.return_value = {1}
    evictor.after_step()
    connector.release_session_cache.assert_called_once_with(
        decode_hashes, shared_hashes
    )
    assert not evictor.has_pending_work

    # The group's release sends the prefix once no live session counts it.
    _finish(manager, scheduler, evictor, second)
    connector.get_pending_save_ids.return_value = set()
    evictor.release(session_ids=[], group_ids=[1])
    assert connector.release_session_cache.call_args.args == (shared_hashes, set())
    assert not evictor.has_pending_work
    evictor.release(session_ids=["session-0", "unknown"], group_ids=[0, 1])
    assert connector.release_session_cache.call_count == 2


def test_a_hash_a_later_prompt_drops_gets_its_cpu_release_hint(manager, scheduler):
    connector = Mock()
    connector.get_pending_save_ids.return_value = set()
    evictor = VLLMSessionKVEvictor(scheduler=scheduler, offload_connector=connector)
    first = _turn(
        manager,
        scheduler,
        evictor,
        "a/turn=0",
        session_id="a",
        group_id=0,
        tokens=list(range(40)),
        output_tokens=list(range(100, 140)),
    )

    # The next prompt keeps the first prompt and rewrites the reply (e.g. stripped reasoning).
    second = _admit(
        scheduler, evictor, "a/turn=1", "a", 0, list(range(40)) + list(range(200, 216))
    )
    kept_hashes = set(second.block_hashes)
    dropped_hashes = set(first.block_hashes) - kept_hashes
    assert dropped_hashes
    connector.release_session_cache.assert_not_called()

    evictor.after_step()
    connector.release_session_cache.assert_called_once_with(dropped_hashes, kept_hashes)


def test_an_idle_sessions_history_outlasts_another_sessions_superseded_gdn_states(
    manager, scheduler, evictor
):
    # vLLM caches a GDN state at each block boundary of d's decode, behind a's needed
    # blocks in the free queue; without the sweep, a's hit is 0 once d holds 466 tokens.
    a = _turn(
        manager,
        scheduler,
        evictor,
        "a/turn=0",
        session_id="a",
        group_id=0,
        tokens=list(range(1000, 1040)),
        output_tokens=list(range(5000, 5010)),
    )
    a_next_turn = list(a.all_token_ids) + [9, 9, 9, 9]
    d = _admit(scheduler, evictor, "d/turn=0", "d", 1, list(range(4000, 4040)))
    _prefill(manager, d)
    evictor.after_step()

    for token in range(6000, 6700):
        _schedule(manager, d, 1, sample=token)
        evictor.after_step()

    assert d.num_tokens == 741
    _, num_hit_tokens, _ = manager.get_computed_blocks(
        _request("a/turn=1", a_next_turn)
    )
    assert num_hit_tokens == 48


def test_a_session_keeps_one_gdn_state_of_its_own_and_its_groups_shared_prompt(
    manager, scheduler, evictor
):
    # Rollouts a and b of one group share a 40-token prompt; its state is at block 1.
    prompt = list(range(40))
    a = _admit(scheduler, evictor, "a/turn=0", "a", 0, prompt)
    _prefill(manager, a, sample=100)
    evictor.after_step()
    b = _admit(scheduler, evictor, "b/turn=0", "b", 0, prompt)
    assert _prefill(manager, b, sample=200) == 32
    evictor.after_step()
    for request, first_token in ((a, 101), (b, 201)):
        for token in range(first_token, first_token + 39):
            _schedule(manager, request, 1, sample=token)
            evictor.after_step()
        _finish(manager, scheduler, evictor, request)

    # Idle: every attention block, the shared prompt's state, and the deepest own state.
    assert _cached_indices(manager, a, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2, 3]
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, a, kv_cache_group_id) == [1, 3]

    # A turn that extends the history hits all of it, even after a step spent waiting.
    turn_1 = _admit(
        scheduler,
        evictor,
        "a/turn=1",
        "a",
        0,
        list(a.all_token_ids) + list(range(300, 310)),
    )
    evictor.after_step()
    assert _prefill(manager, turn_1, sample=400) == 64
    evictor.after_step()
    for token in range(401, 420):
        _schedule(manager, turn_1, 1, sample=token)
        evictor.after_step()
    _finish(manager, scheduler, evictor, turn_1)
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, turn_1, kv_cache_group_id) == [1, 5]

    # A turn that rewrites the last reply resumes from the shared prompt (all states: 80).
    turn_2 = _admit(
        scheduler,
        evictor,
        "a/turn=2",
        "a",
        0,
        turn_1.prompt_token_ids + list(range(500, 530)),
    )
    assert _prefill(manager, turn_2, sample=600) == 32


@pytest.mark.parametrize("b_group_id", [0, 1], ids=["sibling", "other_group"])
def test_an_idle_session_drops_its_prompt_state_once_no_other_session_counts_it(
    manager, scheduler, evictor, b_group_id
):
    # Rollouts a and b share a 40-token prompt (b: a's sibling, or another group's rollout with
    # the same system prompt); its state is at block 1.
    prompt = list(range(40))
    a = _admit(scheduler, evictor, "a/turn=0", "a", 0, prompt)
    _prefill(manager, a, sample=100)
    evictor.after_step()
    b = _admit(scheduler, evictor, "b/turn=0", "b", b_group_id, prompt)
    _prefill(manager, b, sample=200)
    evictor.after_step()
    for request, first_token in ((a, 101), (b, 201)):
        for token in range(first_token, first_token + 39):
            _schedule(manager, request, 1, sample=token)
            evictor.after_step()
        _finish(manager, scheduler, evictor, request)
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, a, kv_cache_group_id) == [1, 3]

    # b ends while a waits on a tool: only a counts the prompt's state now, so it goes, even
    # before the engine steps again.
    evictor.release(session_ids=["b"], group_ids=[])
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, a, kv_cache_group_id) == [3]
    assert _cached_indices(manager, a, ATTENTION_KV_CACHE_GROUP_ID) == [0, 1, 2, 3]


def test_an_idle_session_drops_a_state_its_late_sibling_cached_again(
    manager, scheduler, evictor
):
    # a's sweep drops the 70-token group prompt's state (block 3) while only a counts it. b starts
    # later, prefills the prompt again and caches that state again, below the blocks a's sweeps
    # look at.
    prompt = list(range(70))
    a = _admit(scheduler, evictor, "a/turn=0", "a", 0, prompt)
    _prefill(manager, a, sample=100)
    evictor.after_step()
    for token in range(101, 113):
        _schedule(manager, a, 1, sample=token)
        evictor.after_step()
    b = _admit(scheduler, evictor, "b/turn=0", "b", 0, prompt)
    assert _prefill(manager, b, sample=200) == 0
    evictor.after_step()
    for request, first_token in ((a, 113), (b, 201)):
        for token in range(first_token, first_token + 20):
            _schedule(manager, request, 1, sample=token)
            evictor.after_step()
        _finish(manager, scheduler, evictor, request)
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, a, kv_cache_group_id) == [3, 5]

    # b's release leaves a as the one session counting block 3's state, so it goes.
    evictor.release(session_ids=["b"], group_ids=[])
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, a, kv_cache_group_id) == [5]


def test_a_gdn_block_spanning_several_hashes_keeps_its_deepest_state():
    # 32-token blocks, hashed every 16 tokens: each GDN state sits under every other hash.
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=2 * BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=2 * BLOCK,
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=120,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
        prefix_cache_retention_interval=None,
    )
    manager = KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=2 * BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(kv_cache_manager=manager, requests={})
    evictor = VLLMSessionKVEvictor(scheduler=scheduler)
    request = _admit(scheduler, evictor, "s/turn=0", "s", 0, list(range(100)))
    blocks, hit, _ = manager.get_computed_blocks(request)
    _schedule(manager, request, 96, blocks, hit)
    evictor.after_step()
    _schedule(manager, request, 4, sample=7)
    evictor.after_step()
    for token in range(500, 580):
        _schedule(manager, request, 1, sample=token)
        evictor.after_step()
    _finish(manager, scheduler, evictor, request)

    # All states: hash indices 5 (the prompt's), 7 and 9.
    assert _cached_indices(manager, request, 0) == [1, 3, 5, 7, 9]
    assert _cached_indices(manager, request, 1) == [9]
    next_turn = _request("s/turn=1", list(request.all_token_ids) + [1, 2, 3])
    _, num_hit_tokens, _ = manager.get_computed_blocks(next_turn)
    assert num_hit_tokens == 160


def test_with_mtp_drafts_a_session_keeps_the_state_its_next_turn_hits():
    # vLLM drops the last attention block of a hit when it drafts with MTP, so a turn resumes
    # one block before the deepest state. Keeping the deepest would hit 0 tokens here. The turn
    # also computes that block again and caches a second copy of it, which goes at its end.
    manager = _hybrid_manager(len(MAMBA_KV_CACHE_GROUP_IDS), use_eagle=True)
    scheduler = SimpleNamespace(kv_cache_manager=manager, requests={})
    evictor = VLLMSessionKVEvictor(scheduler=scheduler)
    prompt = list(range(1000, 1040))
    num_hit_tokens = []
    for turn in range(3):
        request = _admit(scheduler, evictor, f"s/turn={turn}", "s", 0, prompt)
        num_hit_tokens.append(
            _prefill(manager, request, sample=5000 + 100 * turn, use_eagle=True)
        )
        evictor.after_step()
        for token in range(5001 + 100 * turn, 5030 + 100 * turn):
            _schedule(manager, request, 1, sample=token)
            evictor.after_step()
        _finish(manager, scheduler, evictor, request)
        computed_block_hashes = request.block_hashes[
            : (request.num_tokens - 1) // BLOCK
        ]
        num_copies = {
            _num_copies(
                manager,
                make_block_hash_with_group_id(block_hash, ATTENTION_KV_CACHE_GROUP_ID),
            )
            for block_hash in computed_block_hashes
        }
        assert num_copies == {1}, turn
        if turn == 0:
            # 70 tokens; all states: blocks 0, 2 and 3.
            for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
                assert _cached_indices(manager, request, kv_cache_group_id) == [2]
        prompt = list(request.all_token_ids) + list(
            range(1100 + 100 * turn, 1120 + 100 * turn)
        )

    assert num_hit_tokens == [0, 48, 96]


def test_with_mtp_drafts_a_state_in_use_at_the_turns_last_sweep_goes_once_free():
    # The deepest state is still in use when the request leaves vLLM (a CPU offload save, or a
    # free that async scheduling defers): the next after_step once it is free evicts it.
    manager = _hybrid_manager(len(MAMBA_KV_CACHE_GROUP_IDS), use_eagle=True)
    scheduler = SimpleNamespace(kv_cache_manager=manager, requests={})
    evictor = VLLMSessionKVEvictor(scheduler=scheduler)
    request = _admit(scheduler, evictor, "s/turn=0", "s", 0, list(range(1000, 1040)))
    _prefill(manager, request, sample=5000, use_eagle=True)
    evictor.after_step()
    for token in range(5001, 5030):
        _schedule(manager, request, 1, sample=token)
        evictor.after_step()
    get = manager.block_pool.cached_block_hash_to_block.get_one_block
    deepest_states = [
        get(make_block_hash_with_group_id(request.block_hashes[3], kv_cache_group_id))
        for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS
    ]
    manager.block_pool.touch(deepest_states)

    # 70 tokens: the next turn resumes from block 2's state; block 3's is in use.
    _finish(manager, scheduler, evictor, request)
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, request, kv_cache_group_id) == [2, 3]

    manager.block_pool.free_blocks(deepest_states)
    evictor.after_step()
    for kv_cache_group_id in MAMBA_KV_CACHE_GROUP_IDS:
        assert _cached_indices(manager, request, kv_cache_group_id) == [2]
    assert evictor._sessions_to_sweep == {}


def test_vllm_kv_cache_capacity_bounds_what_vllm_keeps(manager, scheduler, evictor):
    # Two rollouts of one group share a 40-token prompt, prefilled in one chunk; 3 turns.
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler)),
    )
    scheduler.kv_cache_config = manager.kv_cache_config
    capacity = engine.kv_cache_capacity()
    assert capacity == KVCacheCapacity(
        num_blocks=119,
        attention_block_sizes=(BLOCK,),
        blocks_per_state=3,
        num_idle_states=1,
        num_running_states=3,
    )

    def num_held_blocks() -> int:
        return sum(
            block.ref_cnt > 0 or block.block_hash is not None
            for block in manager.block_pool.blocks[1:]
        )

    histories = [list(range(40)), list(range(40))]
    for turn in range(3):
        requests = [
            _admit(scheduler, evictor, f"{index}/turn={turn}", str(index), 0, history)
            for index, history in enumerate(histories)
        ]
        for step in range(31):
            for index, request in enumerate(requests):
                if step == 0:
                    _prefill(manager, request, sample=100 * turn)
                else:
                    _schedule(manager, request, 1, sample=1000 * index + step)
                evictor.after_step()
                assert num_held_blocks() <= sum(
                    capacity.session_blocks(request.num_tokens, running=True)
                    for request in requests
                )
        for request in requests:
            _finish(manager, scheduler, evictor, request)

        # Idle: the charge counts the shared prompt's 2 blocks per session, and each
        # session's last block, which vLLM caches only once full.
        charged = sum(
            capacity.session_blocks(request.num_tokens, running=False)
            for request in requests
        )
        assert charged - 2 - len(requests) <= num_held_blocks() <= charged
        histories = [
            list(request.all_token_ids)
            + list(range(2000 + 50 * turn, 2010 + 50 * turn))
            for request in requests
        ]
