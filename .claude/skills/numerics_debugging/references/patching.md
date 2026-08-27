# Patching torchtitan to enable capture

The capture is gated on a profiler flag and an `ActivationCaptureProfiler`
constructed inside `Profiler`. Apply these patches before a capture run, then
revert them when you are done (they don't belong on `main`).

## 1. `torchtitan/tools/profiler.py`

Add the import, config field, constructor arg, lifecycle hooks, and builder.

```python
# top of file
from agent_tooling.numerics_debugging.activation_tracer import ActivationCaptureProfiler

# inside Profiler.Config — add next to enable_memory_snapshot
dump_numerics: bool = False
"""Dump per-op activation logs for numerics debugging.
Writes ``{dump_folder}/numerics/rank_{rank}_activations.log``
(per-op stats + norm hashes of inputs / outputs)."""

# Profiler.__init__ — add model kwarg and slots
def __init__(
    self,
    config: "Profiler.Config",
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
    model: torch.nn.Module | None = None,  # for activation capture profiler
) -> None:
    ...
    self.activation_capture_profiler = None
    # ActivationCaptureProfiler registers global module forward hooks on
    # the model so backward ops can recover their owning FQN.
    self._model = model

# Profiler.__enter__ — build the activation capture profiler alongside the memory profiler
self.activation_capture_profiler = self.build_activation_capture_profiler(
    base_folder=self._base_folder,
)

# Profiler.__exit__ — teardown
if self.activation_capture_profiler is not None:
    self.activation_capture_profiler.__exit__(exc_type, exc_val, exc_tb)
    self.activation_capture_profiler = None

# Profiler.step — drive the capture-step cadence
if self.activation_capture_profiler is not None:
    self.activation_capture_profiler.step()

# new method
def build_activation_capture_profiler(self, *, base_folder: str):
    """Create and return an :class:`ActivationCaptureProfiler`, or ``None`` if disabled."""
    cfg = self._config
    if not cfg.dump_numerics or self._model is None:
        return None

    dump_dir = os.path.join(base_folder, "numerics")
    profiler = ActivationCaptureProfiler(
        enabled=True,
        model=self._model,
        dump_dir=dump_dir,
        capture_step=cfg.profile_freq,
    )
    profiler.__enter__()
    return profiler
```

## 2. `torchtitan/trainer.py`

Pass the model to the profiler when entering the training loop:

```python
with config.profiler.build(
    global_step=self.step,
    base_folder=config.dump_folder,
    model=self.model_parts[0],  # add this line
) as profiler:
    ...
```

`model_parts[0]` is the eager model rank-0 owns; that is what
`ActivationCaptureProfiler` installs forward hooks on so DebugMode's ModTracker
can attribute backward ops to the right FQN.

## 3. graph_trainer only — replay traced graph through FQNInterpreter

When the active path is `--compile.mode aot_fx_trace`, the traced graph is
called as `gm(*flat_inputs)`, which bypasses every `nn.Module.forward`. That
means `DebugMode`'s `ModTracker` can no longer attribute ops to a FQN, and
the log degrades to `<none>/op_N_*` everywhere.

The fix is to walk the graph node-by-node with an FX interpreter that
restores the FQN / stack / phase that the traced commit already stashed in
`node.meta`, so capture gets the same context that eager would.

**3a. `torchtitan/experiments/graph_trainer/debug_utils.py`** — append:

```python
class FQNInterpreter(torch.fx.Interpreter):
    """Interpreter that sets activation tracer context vars from node metadata.

    For each node, reads:
    - ``node.meta["custom"]["module_fqn"]`` → sets _current_module_name
    - ``node.meta["stack_trace"]`` → parsed and set as _current_stack_frames
    - ``node.meta["autograd_backward"]`` → sets _current_phase_override

    This is needed because traced graph replay via ``gm(*inputs)`` bypasses
    module forwards entirely, so DebugMode's ModTracker cannot infer the
    module FQN. FQNInterpreter walks the graph node-by-node, restoring the
    context that eager capture would otherwise get from DebugMode and
    autograd metadata.
    """

    def run_node(self, n: torch.fx.Node):
        from contextvars import Token

        from agent_tooling.numerics_debugging.activation_tracer import (
            _current_module_name,
            _current_phase_override,
            _current_stack_frames,
            _parse_stack_trace,
        )

        fqn = (n.meta.get("custom") or {}).get("module_fqn")
        stack_trace = n.meta.get("stack_trace")
        is_backward = n.meta.get("autograd_backward", False)

        phase = "backward" if is_backward else "forward"

        tokens: list[Token] = []
        if fqn:
            tokens.append(_current_module_name.set(fqn))
        if stack_trace:
            tokens.append(_current_stack_frames.set(_parse_stack_trace(stack_trace)))
        tokens.append(_current_phase_override.set(phase))
        try:
            return super().run_node(n)
        finally:
            for token in reversed(tokens):
                token.var.reset(token)
```

`run_traced` already accepts an `interpreter_cls` kwarg (in
`torchtitan/experiments/graph_trainer/make_fx_tracer.py`), so no patch is
needed there — just pass `FQNInterpreter` through.

**3b. `torchtitan/experiments/graph_trainer/trainer.py`** — pick the
interpreter only when capture is active, and thread it through
`run_traced`:

```python
def _maybe_get_fqn_interpreter(self) -> type | None:
    from agent_tooling.numerics_debugging.activation_tracer import (
        is_numerics_capture_active,
    )

    if is_numerics_capture_active():
        from torchtitan.experiments.graph_trainer.debug_utils import FQNInterpreter

        return FQNInterpreter

    return None

# in forward_backward_step, where run_traced is invoked:
outputs = run_traced(
    ...,
    interpreter_cls=self._maybe_get_fqn_interpreter(),
)
```

The interpreter only kicks in on the capture step (and only under
aot_fx_trace), so steady-state training is untouched.
