# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VLLMSessionKVEvictor on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align"
groups with dense retention (as `VLLMEngine` sets it), 16-token blocks standing in for
Qwen3.5's 1,152."""

from types import SimpleNamespace

import pytest
import torch

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
)
from vllm.v1.request import Request

BLOCK = 16
ATTENTION_KV_CACHE_GROUP_ID = 0
MAMBA_KV_CACHE_GROUP_IDS = (1, 2, 3)


@pytest.fixture
def manager(request) -> KVCacheManager:
    """Indirect param: the number of Mamba groups (default 3; 0 is a dense model)."""
    num_mamba_groups = getattr(request, "param", len(MAMBA_KV_CACHE_GROUP_IDS))
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
    )


@pytest.fixture
def scheduler(manager) -> SimpleNamespace:
    """The vLLM scheduler attributes the evictor reads."""
    return SimpleNamespace(kv_cache_manager=manager, requests={})


@pytest.fixture
def evictor(scheduler) -> VLLMSessionKVEvictor:
    return VLLMSessionKVEvictor(scheduler=scheduler)


def test_kv_cache_groups_with_blocks_larger_than_the_hash_block_raise(scheduler):
    # vLLM would cache partial blocks under extra hashes (aliases), which the evictor misses.
    scheduler.kv_cache_manager.block_pool.hash_block_size = BLOCK // 2
    with pytest.raises(ValueError, match="hash_block_size 8"):
        VLLMSessionKVEvictor(scheduler=scheduler)


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


def _prefill(manager, request, sample: int = 7) -> int:
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
    evictor.track(
        request_id, request_id=request_id, session_id=session_id, group_id=group_id
    )
    return request


def _finish(manager, scheduler, evictor, request) -> None:
    manager.free(request)
    del scheduler.requests[request.request_id]
    evictor.finish([request.request_id])


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
    assert evictor._num_owners_by_hash == {}


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
    assert evictor._session_ids_by_group == {2: {"group=2/rollout=0"}}
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


def test_a_dead_copy_of_a_hash_is_evicted_while_a_running_request_holds_another(
    manager, scheduler, evictor
):
    # vLLM may cache one prefix in two blocks. Rollout b's request holds the first copy and runs;
    # rollout a computed the same prompt again (no prefix hit) and finished: its copy is free.
    prompt = list(range(40))  # 2 full blocks
    running = _admit(scheduler, evictor, "b/turn=0", "b", 2, prompt)
    _prefill(manager, running)
    copy = _admit(scheduler, evictor, "a/turn=0", "a", 1, prompt)
    _schedule(manager, copy, len(prompt), sample=7)
    _finish(manager, scheduler, evictor, copy)

    # Both rollouts end while b's request still runs.
    evictor.release(session_ids=["a", "b"], group_ids=[])

    # a's copy goes now; only blocks that b's request holds stay cached.
    blocks = manager.block_pool.blocks
    cached = [block for block in blocks if block.block_hash is not None]
    assert cached and all(block.ref_cnt > 0 for block in cached)


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


def test_occupancy_counts_live_rollouts_cached_blocks_but_not_ended_ones(
    manager, scheduler, evictor
):
    num_usable_blocks = manager.block_pool.num_gpu_blocks - 1  # minus vLLM's null block
    assert evictor.kv_cache_reading().occupancy == 0.0
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
    ended = _turn(
        manager,
        scheduler,
        evictor,
        "ended/turn=0",
        session_id="ended",
        group_id=1,
        tokens=list(range(500, 550)),
        output_tokens=list(range(600, 620)),
    )
    num_live_blocks = _num_cached_blocks(manager, live)
    num_ended_blocks = _num_cached_blocks(manager, ended)
    reading = evictor.kv_cache_reading()
    # Nothing runs, so vLLM's usage reads 0; both rollouts' cached blocks count here.
    assert reading.usage == 0.0
    assert reading.occupancy == pytest.approx(
        (num_live_blocks + num_ended_blocks) / num_usable_blocks
    )

    evictor.release(session_ids=["ended"], group_ids=[1])
    assert evictor.kv_cache_reading().occupancy == pytest.approx(
        num_live_blocks / num_usable_blocks
    )
