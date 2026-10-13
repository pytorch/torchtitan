# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by the MoE paged stash tests: the runtime's
(``test_paged_stash.py``), the eager frontend's (``test_paged_stash_eager.py``)
and GraphTrainer's (``experiments/graph_trainer/tests/test_paged_stash.py``).

A frontend stashes and reloads its own way -- the runtime's tests call the
manager, GraphTrainer's call its custom ops -- so the round-trip helpers take
the ``stash`` and ``reload`` to use, defaulting to the manager's.
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    RECV_F,
    SEND_F,
)

from torchtitan.components.optim.optimizer import AdamW, OptimizersContainer
from torchtitan.distributed.deepep.paged_stash.runtime import (
    PagedStashManager,
    StashSlot,
    wait_for_copy,
    warmup_paged_stash_kernels,
)

BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
MAX_TOKENS = 4096  # capacity-padded rows per activation: the worst case
HIDDEN = 256
PAGE_SIZE = 64
KEY = (BF16, HIDDEN)


class StashFn(Protocol):
    def __call__(
        self,
        slot: StashSlot,
        tensor: torch.Tensor,
        num_tokens: torch.Tensor,
        *,
        prefetch: int = 0,
    ) -> torch.Tensor:
        """Stash ``tensor``; returns its handle."""
        ...


ReloadFn = Callable[[StashSlot, torch.Tensor, torch.Tensor, torch.Tensor], Any]
"""``reload(slot, handle, num_tokens, like)``: the activation ``handle`` holds,
shaped like ``like``."""


@contextlib.contextmanager
def fresh_manager() -> Iterator[PagedStashManager]:
    """The process-wide manager, reset and enabled on cuda:0."""
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.page_size = PAGE_SIZE
    try:
        yield manager
    finally:
        PagedStashManager.reset_instance()


def make_slots(
    manager: PagedStashManager,
    count: int,
    *,
    dtype: torch.dtype = BF16,
    token_divisor: int = 1,
    max_tokens: int = MAX_TOKENS,
    stage_index: int = 0,
) -> list[StashSlot]:
    """One slot per layer, as a frontend registers them."""
    slots = []
    for layer in range(count):
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=dtype,
            hidden_size=HIDDEN,
            max_num_tokens=max_tokens,
            token_divisor=token_divisor,
            layer_id=layer,
            node_name=f"s{stage_index}l{layer}",
            stage_index=stage_index,
        )
        manager.register_slot(slot)
        slots.append(slot)
    return slots


def pages(rows: int) -> int:
    return -(-rows // PAGE_SIZE)


def random_activation(slot: StashSlot) -> torch.Tensor:
    """A padded activation of random bit patterns; flat for block scales."""
    itemsize = torch.empty((), dtype=slot.dtype).element_size()
    shape = (slot.max_stash_tokens, HIDDEN * itemsize)
    rows = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    rows = rows.view(slot.dtype)
    return rows.reshape(-1) if slot.token_divisor > 1 else rows


def live_bytes(tensor: torch.Tensor, slot: StashSlot, num_tokens: int) -> torch.Tensor:
    """The bytes of the rows ``num_tokens`` live tokens occupy."""
    rows = tensor.reshape(-1, slot.hidden_size).view(torch.uint8)
    return rows[: num_tokens // slot.token_divisor]


def new_handle(slot: StashSlot) -> torch.Tensor:
    """A handle as the frontends size it: page ids and a spill flag. It starts
    as -1s, so whatever a stash does not write shows."""
    return torch.full(
        (pages(slot.max_stash_tokens) + 1,), -1, dtype=torch.int64, device="cuda"
    )


def stash(
    slot: StashSlot,
    tensor: torch.Tensor,
    num_tokens: torch.Tensor,
    *,
    prefetch: int = 0,
) -> torch.Tensor:
    """Page ``tensor`` out through the manager. Returns its handle."""
    handle = new_handle(slot)
    PagedStashManager.get_instance().stash(
        slot, tensor, num_tokens, handle=handle, prefetch=prefetch
    )
    return handle


def reload(
    slot: StashSlot,
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    """Page the activation ``handle`` holds back in through the manager, and
    join its copy."""
    del num_tokens, like  # the stash record holds them
    manager = PagedStashManager.get_instance()
    out = manager.materialize(manager.take_stash_record(slot, handle))
    wait_for_copy(out)
    return out


def round_trip(
    slots: list[StashSlot],
    tensors: list[torch.Tensor],
    counts: list[torch.Tensor],
    *,
    stash: StashFn = stash,
    reload: ReloadFn = reload,
) -> list[torch.Tensor]:
    """Stash every activation, then reload them all: one forward/backward."""
    handles = [stash(*args) for args in zip(slots, tensors, counts, strict=True)]
    return [reload(*args) for args in zip(slots, handles, counts, tensors, strict=True)]


def assert_live_rows_equal(slots, expected, actual, live) -> None:
    for slot, want, got, n in zip(slots, expected, actual, live, strict=True):
        assert torch.equal(live_bytes(got, slot, n), live_bytes(want, slot, n))


def setup_stash(
    manager: PagedStashManager,
    slots: list[StashSlot],
    live: list[int],
    *,
    stash: StashFn = stash,
    reload: ReloadFn = reload,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Take ``slots`` through the measure step with ``live`` tokens each.

    Returns the activations, their counts, and what the measure step's reloads
    handed back; the buffers are allocated and their kernels compiled.
    """
    tensors = [random_activation(slot) for slot in slots]
    counts = [torch.tensor(n, dtype=torch.int64, device="cuda") for n in live]
    manager.begin_step()
    assert not manager.active, "the first step must run in measure mode"
    measured = round_trip(slots, tensors, counts, stash=stash, reload=reload)
    manager.begin_step()
    assert manager.active, "buffers must exist after the measure step"
    warmup_paged_stash_kernels()
    return tensors, counts, measured


def capture_round_trip(
    manager: PagedStashManager,
    slots: list[StashSlot],
    tensors: list[torch.Tensor],
    counts: list[torch.Tensor],
    *,
    stash: StashFn = stash,
    reload: ReloadFn = reload,
) -> tuple[list[torch.Tensor], Callable[[list[int]], None]]:
    """Capture one round trip in a CUDA graph.

    Returns what the graph reloads into, and a function that replays it on
    fresh activations with the given live token counts.
    """
    manager.begin_step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = round_trip(slots, tensors, counts, stash=stash, reload=reload)

    def replay(live: list[int]) -> None:
        for slot, tensor, count, n in zip(slots, tensors, counts, live, strict=True):
            tensor.copy_(random_activation(slot))
            count.fill_(n)
        manager.begin_step()
        graph.replay()
        torch.cuda.synchronize()

    return outputs, replay


def actions(spec: str) -> list[_Action]:
    """``"F0.1 B1.0"``: forward of stage 0 microbatch 1, then full backward of
    stage 1 microbatch 0. ``I``/``W`` split the backward into its input and
    weight halves; ``S``/``R`` are forward sends and receives."""
    kinds = {
        "F": FORWARD,
        "B": FULL_BACKWARD,
        "I": BACKWARD_INPUT,
        "W": BACKWARD_WEIGHT,
        "S": SEND_F,
        "R": RECV_F,
    }
    result = []
    for token in spec.split():
        stage, microbatch = token[1:].split(".")
        result.append(_Action(int(stage), kinds[token[0]], int(microbatch)))
    return result


# AdamW implementations, as their AdamW config fields.
ADAMW_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "fused": {"fused": True},
    "fused_bf16_moments": {"fused": True, "moment_dtype": "bfloat16"},
    "foreach": {"fused": False, "foreach": True},
    "for_loop": {"fused": False, "foreach": False},
}


def optimizers(model: nn.Module, implementation: str = "fused") -> OptimizersContainer:
    """AdamW over every parameter of ``model``."""
    config = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(pattern=r".*", lr=0.1, **ADAMW_IMPLEMENTATIONS[implementation])
        ]
    )
    return config.build(model_parts=[model])


def drains_the_stream(fn: Callable[[], Any]) -> tuple[Any, bool]:
    """Call ``fn`` behind ~1.7s of queued GPU work.

    Returns its result and whether that work had drained by the time ``fn``
    returned, which is what a host synchronization inside ``fn`` would cause.
    """
    torch.cuda.synchronize()
    end = torch.cuda.Event()
    torch.cuda._sleep(2_000_000_000)
    result = fn()
    end.record()
    drained = end.query()
    torch.cuda.synchronize()
    return result, drained
