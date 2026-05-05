# TorchTitan Observability

Structured logging for distributed training. Emits per-rank JSONL events for phase timing, diagnostics, and post-hoc analysis.

Design principles:
- LLM friendly: Emit structured, per-rank data that can be queried during and after a run.
- Handle both SPMD pretraining (one process group) and RL with multiple independent actors (no shared process group).
- Stay invisible to users -- no metric dictionaries to pass around.
- Never block training.
- Support pluggable backends via handler factories.

## Quickstart

```python
from torchtitan.tools.logging import init_logger
from torchtitan.observability import structured_logger as sl

# console logger (stdout, [titan] prefix)
init_logger()

# Register handlers, i.e. the functions that will take the logs
# and save to a local jsonl, a database, or something else.
sl.init_structured_logger(source="training", output_dir="./outputs")

# Use to register a point-in-time marker — training has started.
sl.log_trace_instant("training_start")

loaded_step = 0
for step in range(loaded_step + 1, num_steps + 1):
    # Stamp every subsequent record with `step` and `relative_step`
    sl.set_step(step, relative_step=step - loaded_step)

    if should_garbage_collect:
        # Appends "gc" to `step_tags` on every record for this step; tags
        # reset at the next set_step() call.
        # Users can filter such steps later, e.g. "ignore if tag X".
        sl.add_step_tag("gc")
        with sl.log_trace_span("gc_collect"):
            run_gc()

    with sl.log_trace_span("fwd_bwd"):
        output = model(batch)
        loss.backward()

    with sl.log_trace_span("Optimizer"):
        optimizer.step()

    # Scalars you may want to register to help debug, for example.
    sl.log_trace_scalar({
        "num_trainable_tokens": num_trainable_tokens,
         "batch_size": bsz
         })
```

Call `init_logger()` and `sl.init_structured_logger()` once per process before any trace calls. Rank and source are baked into the formatter at init; every JSONL entry automatically includes `rank`, `source`, `caller` (file:line:function), `time_us`, `step`, `relative_step`, and (when tags are set) `step_tags`.

## API reference

See docstrings for full args:

- `sl.init_structured_logger(source, output_dir, rank=None, enable=True)` -- wire up handlers; call once per process before any trace call. Pass ``enable=False`` (or set ``--debug.enable_structured_logging=False``) to make all trace calls no-ops.
- `sl.log_trace_span(event_type, description=None, *, stacklevel=2)` -- context manager / decorator; emits `_start` / `_end` / optional `_error` records.
- `sl.log_trace_instant(event_type, *, stacklevel=2)` -- point-in-time marker (no duration).
- `sl.log_trace_scalar(scalars, *, stacklevel=2)` -- emit `metric_value` records from a `{name: number}` dict.
- `sl.set_step(step, *, relative_step=None)` -- stamp subsequent records with a step; clears previous step's tags.
- `sl.add_step_tag(tag)` / `sl.clear_step_tags()` -- annotate the current step (e.g. `"gc"`, `"eval"`). `clear_step_tags` is called at `set_step`.
- `TITAN_STRUCT_LOGGER_HANDLERS` -- Define handlers at the env level

## Flow of information

End-to-end, what happens when user code calls one of the `sl.` helpers:

```
user code
    │   with sl.log_trace_span("fwd_bwd"):
    │       ...
    │
    │   # On entry, log_trace_span calls:
    │       _structured_logger.info(
    │           msg="[step 5] fwd_bwd_start",       ← registers human-readable string
    │           extra=event_extra(                  ← structured payload
    │               event_type="fwd_bwd_start",     ← → record.log_type_name
    │               step=5,                         ← → record.step
    │               task_name=None,                 ← → record.task_name
    │           ),
    │       )
    │
    │   # sl.set_step() / sl.add_step_tag() write into a ContextVar and a
    │   # module global; get_step() / get_step_tags() read them from inside
    │   # event_extra(...) so every record picks up the current step + tags.
    ▼
_structured_logger  (logging.Logger, name="torchtitan.structured_logger",
                     propagate=False — records stay out of the root logger)
    │
    ▼
TraceEventsOnlyFilter  (drops records that reached the logger WITHOUT a
                        log_type_name attribute — defensive; shouldn't fire
                        in practice because only the sl.* helpers write here)
    │
    ├── TraceJsonlHandler      ──▶  TraceJsonlFormatter     ──▶  {output_dir}/structured_logs/*.jsonl
    └── TraceMyDBHandler*     ──▶  TraceMyDBFormatter     ──▶  MyDB # extra handler defined by user
```

## Custom handlers

`TITAN_STRUCT_LOGGER_HANDLERS` is a comma-separated list of fully-qualified Python function paths. When set, ONLY the listed factories run.

```bash
export TITAN_STRUCT_LOGGER_HANDLERS="torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler,mypackage.my_backend.register_my_db_handler"
```

A handler factory takes the args `sl.init_structured_logger()` forwards and attaches one handler to `structured_logger`. Example: stream events to a remote database instead of writing to disk, and enrich each record with cluster metadata along the way.

```python
import logging
import os

from torchtitan.observability.structured_logger.jsonl_handler import (
    TraceJsonlFormatter,
)
from torchtitan.observability.structured_logger.structured_logging import (
    TraceEventsOnlyFilter,
)


class MyDBFormatter(TraceJsonlFormatter):
    """Enrich each record with backend-specific fields before serialization."""

    def _log_dict(self, record):
        d = super()._log_dict(record)
        d["cluster"] = os.environ.get("CLUSTER_NAME", "unknown")
        return d


class MyDBHandler(logging.Handler):
    """Send each trace event to a remote DB as it's emitted."""

    def __init__(self, rank: int, source: str, db_url: str):
        super().__init__()
        self.client = MyDBClient(db_url)
        self.setFormatter(MyDBFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.client.insert_row(self.format(record))
        except Exception:
            self.handleError(record)


def register_my_db_handler(
    *, structured_logger: logging.Logger, rank: int, source: str, **kw
) -> None:
    structured_logger.addHandler(
        MyDBHandler(rank=rank, source=source, db_url="mydb://..."),
    )
```

## Analysis

### Gantt trace from JSONL

```python
from torchtitan.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
)

generate_gantt_trace("outputs/structured_logs/", "outputs/analysis/gantt.json")
```

Reads every rank's JSONL, pairs `_start` / `_end`, and writes a Chrome Trace JSON. Open in [Perfetto](https://ui.perfetto.dev).

Example dummy ASCII sketch:
```
                       step 1                              step 2
                  ┌──────────────────────────┐       ┌───────────────────────────┐
trainer rank 0    │ ▓▓ fwd_bwd ▓▓  ▓ optim ▓ │       │ ▓ fwd_bwd ▓▓ ▓▓ optim ▓   │
trainer rank 1    │  ▓▓ fwd_bwd ▓ ▓ optim ▓  │       │  ▓▓ fwd_bwd ▓▓ ▓ optim ▓  │
trainer rank 2    │ ▓ fwd_bwd ▓▓ ▓ optim ▓▓  │       │  ▓ fwd_bwd ▓▓▓ ▓ optim ▓▓ │
trainer rank 3    │  ▓▓ fwd_bwd ▓▓▓ ▓ optim ▓│       │ ▓ fwd_bwd ▓▓ ▓ optim ▓    │
                  └──────────────────────────┘       └───────────────────────────┘
controller        ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓▓ ▓ scoring ▓ ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓
rollouter         ▓▓▓                                     ▓▓▓
reward                                       ▓▓▓                              ▓▓▓
                  ├──────────────────────────┼───────────┼──────────────────────┤
                  0s                         3s          4s                     7s
```

Each `log_trace_span` becomes a bar. Every source file becomes a separate process row. Auto-named asyncio tasks are slot-packed into a contiguous `0..K-1` range where K is peak concurrency, so the track count stays readable even for long async runs (e.g. RL actors dispatching many concurrent RPCs).
