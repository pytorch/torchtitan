# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step state for structured trace records.

Stores current step, relative step, and step tags. Every trace call
(``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar``)
stamps records with these values at emit time.

Two runtime models share this state:

- **SPMD trainer.** One process, one thread, no asyncio. Call
  ``set_step(step)`` once per step; add tags like ``add_step_tag("gc")``
  when GC ran. A module global suffices.
- **Async RL.** Multiple actor tasks may run concurrently on the same
  process (e.g., one actor doing GC while another runs evaluation).
  Each task may add DIFFERENT tags for the same step, and those views
  must stay isolated -- actor A's ``"gc"`` must not leak into actor B's
  view of ``"eval"``.

Design
------
Each value is stored in two places:

- **module-level global** -- SPMD path; also the fallback for plain
  threads (``threading.Thread``), which don't inherit ContextVar state.
- **ContextVar** -- per-task path; inherited when an asyncio task is
  spawned, isolated from sibling tasks.

``set_step`` writes both (same value -- steps agree across tasks).
``add_step_tag`` is task-aware: inside an asyncio task it writes the
CV (ContextVar) only; outside, the global only. This prevents cross-pollination
when sibling actor tasks tag independently. Readers check the CV
first, fall back to the global.

Example::

    # Pre-asyncio SPMD code sets a global tag (e.g. from init logic):
    add_step_tag("checkpoint_step")                 # writes _TAGS_GLOBAL
    assert get_step_tags() == ("checkpoint_step",)  # SPMD path reads global

    # Once inside async tasks, tags are task-scoped:
    set_step(42)
    async def actor_gc():
        add_step_tag("gc")                          # writes CV only
        return get_step_tags()
    async def actor_eval():
        add_step_tag("eval")                        # writes CV only
        return get_step_tags()

    gc_view, eval_view = await asyncio.gather(actor_gc(), actor_eval())
    assert gc_view == ("gc",)                       # actor A: just its CV
    assert eval_view == ("eval",)                   # actor B: just its CV
    assert _TAGS_GLOBAL == ("checkpoint_step",)     # global untouched

Note: inside a task, global tags are NOT visible (``get_step_tags`` returns
the task's CV when non-empty). Actor contexts are intentionally isolated;
if a tag needs to reach actors, set it per-task (e.g. via the controller's
sync_step broadcast).
"""

import asyncio
from contextvars import ContextVar

# Module-level globals. Source of truth for SPMD and non-asyncio reads.
# Tuples (not lists): immutable, so snapshots can't be mutated-through-
# reference to bypass CV scope-isolation. Concurrent tasks don't overwrite
# each other -- asyncio copies the context at task spawn, giving each its
# own CV copy; writes are scope-local.
_STEP_GLOBAL: int | None = None
_RELATIVE_STEP_GLOBAL: int | None = None
_TAGS_GLOBAL: tuple[str, ...] = ()

# Per-task overrides. Async endpoints (e.g. RL actor frameworks) run on fresh
# asyncio tasks that inherit their parent's Context via asyncio.create_task's
# implicit copy_context(); writes stay scoped to the writing task.
_STEP_CV: ContextVar[int | None] = ContextVar("_STEP_CV", default=None)
_RELATIVE_STEP_CV: ContextVar[int | None] = ContextVar(
    "_RELATIVE_STEP_CV", default=None
)
_TAGS_CV: ContextVar[tuple[str, ...]] = ContextVar("_TAGS_CV", default=())


def _is_in_async_task() -> bool:
    """True iff called from inside a running asyncio task.

    On Python 3.12 ``asyncio.current_task()`` calls
    ``get_running_loop()`` which raises ``RuntimeError`` when no loop is
    running. On Python 3.14+ it's expected to return None directly. The
    try/except handles both.
    """
    try:
        return asyncio.current_task() is not None
    except RuntimeError:
        return False


def set_step(step: int, *, relative_step: int | None = None) -> None:
    """Set the current training step. All subsequent trace records will
    include this step number. Clears step tags from the previous step.

    Args:
        step: Absolute training step (e.g., 12345 after resuming).
        relative_step: Steps since process start (1 on first step of each
            restart). Used by downstream analysis tools to align
            per-restart event timelines. When None, defaults to ``step``
            -- correct for runs without checkpoint resume. Resumed
            trainers must pass it explicitly.
    Example::
        self.checkpointer.load(step=config.checkpoint.load_step)
        loaded_step = self.step
        for step in range(loaded_step + 1, num_steps + 1):
            set_step(step, relative_step=step - loaded_step)
            train_step(...)
    """
    global _STEP_GLOBAL, _RELATIVE_STEP_GLOBAL
    if relative_step is None:
        relative_step = step
    _STEP_GLOBAL = step
    _STEP_CV.set(step)
    _RELATIVE_STEP_GLOBAL = relative_step
    _RELATIVE_STEP_CV.set(relative_step)
    clear_step_tags()


def get_step() -> int | None:
    """Return the current step, or None if not set."""
    cv = _STEP_CV.get()
    if cv is not None:
        return cv
    return _STEP_GLOBAL


def get_relative_step() -> int | None:
    """Return the relative step (steps since process start), or None."""
    cv = _RELATIVE_STEP_CV.get()
    if cv is not None:
        return cv
    return _RELATIVE_STEP_GLOBAL


def get_step_tags() -> tuple[str, ...]:
    """Return the current step tags (CV if non-empty, else global)."""
    cv = _TAGS_CV.get()
    if cv:
        return cv
    return _TAGS_GLOBAL


def add_step_tag(tag: str) -> None:
    """Annotate the current step. Tags appear in trace JSONL for filtering.

    Task-aware write: inside an asyncio task the tag is appended to the
    CV only (so sibling tasks with different tags don't cross-pollinate
    via the shared global); outside any task the tag goes to the global
    only (SPMD path).

    Example::

        if gc_happened:
            add_step_tag("gc")
        if is_validation:
            add_step_tag("eval")
    """
    global _TAGS_GLOBAL
    if _is_in_async_task():
        current = _TAGS_CV.get()
        if tag not in current:
            _TAGS_CV.set(current + (tag,))
    else:
        if tag not in _TAGS_GLOBAL:
            _TAGS_GLOBAL = _TAGS_GLOBAL + (tag,)


def clear_step_tags() -> None:
    """Reset step tags."""
    global _TAGS_GLOBAL
    _TAGS_GLOBAL = ()
    _TAGS_CV.set(())
