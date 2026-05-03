# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight activation tracer using monkeypatch + TorchDispatchMode.

Captures intermediate tensor values during training without modifying
model source code. Works with DTensor and torch.compile.

For each captured op, we track:
    - **Op name**: the ATen operation (e.g. mm, bmm, silu, add)
    - **Module FQN**: which module the op belongs to
      (e.g. ``layers.0.attention.qkv_linear.wq``)
    - **Source location**: stack trace back to the model code that
      triggered the op (e.g. ``common/attention.py:534``)
    - **Phase**: whether the op ran during forward or backward

What gets captured:
    By default, captures outputs of common compute ops — matrix multiplies
    (mm, bmm, addmm), activations (silu, gelu, relu, softmax), element-wise
    ops (mul, add), and normalization (layer_norm, rms_norm). These are the
    ops whose numerics matter for training parity.

    Skips infrastructure ops (view, reshape, transpose, clone, copy_, detach,
    collective comms) and small tensors (< 1000 elements) since they don't
    carry meaningful numerics information.

    Each captured tensor is keyed by ``module_fqn/op_N_opname`` (e.g.
    ``layers.0.attention.qkv_linear.wq/op_0_mm``) and tagged with its
    phase (forward or backward).

    See DEFAULT_COMPUTE_OPS and _EXCLUDED_OPS for the full lists.
    See CAPTURED_OPS.md for details and customization examples.
"""

import functools
import os
import re
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass
class _StackFrame:
    filename: str
    lineno: int
    name: str
    line: str | None = None

    def short_str(self) -> str:
        parts = self.filename.replace("\\", "/").split("/")
        short_path = "/".join(parts[-2:]) if len(parts) >= 2 else self.filename
        return f"{short_path}:{self.lineno}"


@dataclass
class CapturedActivation:
    """A captured activation with its source location."""

    tensor: torch.Tensor
    stack_frames: list[_StackFrame] = field(default_factory=list)
    phase: str = "forward"

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype


# Ops to always exclude — infrastructure, not meaningful for numerics parity.
_EXCLUDED_OPS = frozenset(
    {
        "empty",
        "zeros",
        "ones",
        "full",
        "copy_",
        "_to_copy",
        "to",
        "clone",
        "contiguous",
        "view",
        "reshape",
        "permute",
        "transpose",
        "unsqueeze",
        "squeeze",
        "expand",
        "repeat",
        "cat",
        "stack",
        "split",
        "chunk",
        "narrow",
        "select",
        "index_select",
        "gather",
        "scatter",
        "slice",
        "as_strided",
        "detach",
        "redistribute",
        "all_gather",
        "all_reduce",
        "reduce_scatter",
    }
)

DEFAULT_COMPUTE_OPS = frozenset(
    {
        "mm",
        "matmul",
        "bmm",
        "addmm",
        "linear",
        "einsum",
        "conv",
        "softmax",
        "layer_norm",
        "rms_norm",
        "gelu",
        "silu",
        "relu",
        "mul",
        "add",
    }
)


def _is_filtered_frame(path: str) -> bool:
    """Filter frames from torch internals and activation_tracer.

    Used by both _extract_user_frames (live traceback) and
    _parse_stack_trace (node.meta / autograd traceback strings).
    """
    if "/torch/nn/" in path or "\\torch\\nn\\" in path:
        return True
    if "/torch/autograd/" in path or "\\torch\\autograd\\" in path:
        return True
    if "activation_tracer.py" in path:
        return True
    return False


def _extract_user_frames(stack: list[traceback.FrameSummary]) -> list[_StackFrame]:
    """Extract user-code frames from a live traceback, filtering out
    torch internals and activation_tracer frames."""
    frames = []
    for frame in reversed(stack):
        if not _is_filtered_frame(frame.filename) and not frame.filename.startswith("<"):
            frames.append(
                _StackFrame(
                    filename=frame.filename,
                    lineno=frame.lineno,
                    name=frame.name,
                    line=frame.line,
                )
            )
    return frames


def _get_op_name(func) -> str:
    if hasattr(func, "_schema"):
        return func._schema.name.split("::")[-1]
    return str(func).split(".")[-1].replace(">", "")


def _should_capture(func, result, min_numel: int, op_filter: set[str] | None) -> bool:
    """Filter to only capture "interesting" tensors.

    Filters by:
    - Tensor type (must be a real tensor, not FakeTensor)
    - Size (must have at least min_numel elements)
    - Dtype (must be float32, float16, or bfloat16)
    - Always excludes infrastructure ops (empty, copy_, view, etc.)
    - Optionally by op name (if op_filter is provided)

    For DTensors, the local tensor is used for size/dtype checks.

    Args:
        func: The dispatch function.
        result: The operation result.
        min_numel: Minimum number of elements to capture.
        op_filter: Optional set of op name substrings to match.
                  If None, captures all ops that pass other filters.
                  Use DEFAULT_COMPUTE_OPS for common compute ops only.

    Returns:
        True if the result should be captured.
    """
    if not isinstance(result, torch.Tensor):
        return False

    # Skip FakeTensors produced during make_fx tracing.
    if isinstance(result, torch._subclasses.FakeTensor):
        return False

    try:
        from torch.distributed.tensor import DTensor

        tensor = result._local_tensor if isinstance(result, DTensor) else result
    except ImportError:
        tensor = result

    if tensor.numel() < min_numel:
        return False
    if tensor.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        return False

    op_name = _get_op_name(func)
    if op_name in _EXCLUDED_OPS:
        return False
    if op_filter is None:
        return True
    return any(f in op_name for f in op_filter)


# The current module FQN for the op being dispatched.
#
# Set by _patch_model's wrapped_forward during eager execution, and by
# FQNInterpreter.run_node during traced graph replay (from
# node.meta["custom"]["module_fqn"]).  Read by _maybe_capture to build
# the capture key (e.g. "layers.0.attention.wq/op_0_mm").
#
# During eager backward, the monkeypatched forwards don't fire (C++
# autograd engine bypasses them), so this is None.  _maybe_capture
# falls back to _grad_fn_to_module to recover the FQN from the
# autograd graph.
_current_module_name: ContextVar[str | None] = ContextVar(
    "_current_module_name", default=None
)

# Override for stack frames attached to captured activations.
#
# Set by _patch_model (from inspect.getfile of the module's forward
# method) during eager forward, and by FQNInterpreter (from
# node.meta["stack_trace"]) during traced replay.  When set,
# _maybe_capture uses these instead of the live traceback (which is
# uninformative under FSDP's C++ dispatch or during traced replay).
#
# During eager backward, this is None and _maybe_capture reads the
# forward traceback from _current_autograd_node().metadata["traceback_"]
# (enabled by detect_anomaly).
_current_stack_frames: ContextVar[list[_StackFrame] | None] = ContextVar(
    "_current_stack_frames", default=None
)

# Phase override for the current op ("forward" or "backward").
#
# Set by FQNInterpreter from node.meta["autograd_backward"] during
# traced replay.  When set, _maybe_capture uses this directly instead
# of inferring the phase from the autograd state.
#
# None means no override (eager mode) — _maybe_capture falls back to
# the _in_backward sticky flag on the dispatch mode.
_current_phase_override: ContextVar[str | None] = ContextVar(
    "_current_phase_override", default=None
)

# Global flag indicating that NumericsDebugger has armed capture.
#
# Checked by GraphTrainer._get_interpreter_cls() to decide whether to
# route traced graph replay through FQNInterpreter (which populates
# _current_module_name from node metadata) instead of direct gm(*inputs).
# Module-level bool rather than a callback — each rank is a separate
# process, so no thread-safety concern.
_numerics_capture_active = False


def is_numerics_capture_active() -> bool:
    return _numerics_capture_active


def set_numerics_capture_active(active: bool) -> None:
    global _numerics_capture_active
    _numerics_capture_active = active


_STACK_TRACE_RE = re.compile(
    r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*?)(?=\n|$)'
)


def _parse_stack_trace(trace_str: str) -> list[_StackFrame]:
    """Parse File/line/func entries from node.meta["stack_trace"],
    filtering out torch.nn.modules frames."""
    frames = []
    for m in _STACK_TRACE_RE.finditer(trace_str):
        if not _is_filtered_frame(m.group(1)):
            frames.append(
                _StackFrame(
                    filename=m.group(1),
                    lineno=int(m.group(2)),
                    name=m.group(3),
                    line=m.group(4).strip() or None,
                )
            )
    return frames


def _clean_fqn(fqn: str) -> str:
    """Strip activation-checkpoint wrapper names from FQNs."""
    return fqn.replace("._checkpoint_wrapped_module", "")


def _resolve_module_location(module: nn.Module) -> list[_StackFrame]:
    """Resolve source location from the module class definition.

    Under FSDP and activation checkpointing, module forwards are
    dispatched from C++ hooks, so the Python call stack inside
    ``__torch_dispatch__`` only shows the top-level entry point
    (e.g. ``train.py:main``).  The model-level frames
    (``attention.py``, ``decoder.py``) are not on the stack because
    the C++ runtime broke the Python call chain.

    We work around this by using ``inspect.getfile`` to look up
    where the module's ``forward`` method is defined at patch time.
    This gives a stable location (e.g. ``common/attention.py:278``)
    regardless of how the module is invoked at runtime.

    For aot_fx_trace mode, ``node.meta["stack_trace"]`` provides
    exact per-op line numbers recorded during tracing, so this
    fallback is only needed for eager mode.
    """
    import inspect

    try:
        fn = type(module).forward
        source_file = inspect.getfile(fn)
        if _is_filtered_frame(source_file):
            return []
        _, lineno = inspect.getsourcelines(fn)
        return [_StackFrame(filename=source_file, lineno=lineno, name="forward")]
    except (TypeError, OSError):
        return []


# Maps autograd Node (grad_fn) to module FQN. Built during forward by
# _patch_model, read during backward by _maybe_capture to give backward
# ops their originating module's FQN.
_grad_fn_to_module: dict[object, str] = {}


def _patch_model(model: nn.Module) -> list[tuple[nn.Module, callable]]:
    """Monkeypatch every module's forward to track module context.

    Three things happen in each wrapped forward:

    1. Set _current_module_name so _maybe_capture knows which module
       the current op belongs to (used as the key prefix).
    2. Set _current_stack_frames to the module's source location
       (from inspect.getfile) so captured ops get a meaningful
       location even when the live traceback is empty (FSDP).
    3. After forward returns, walk the autograd graph from the output
       grad_fn to build the _grad_fn_to_module mapping.  This lets
       _maybe_capture recover the module FQN for backward ops, where
       _current_module_name is not set (C++ autograd engine bypasses
       the monkeypatched forwards).

    The default args (_name, _orig, _loc) capture loop variables
    by value — without them, all closures would share the last
    iteration's values.

    Returns a list of (module, original_forward) pairs for unpatching.
    """
    _grad_fn_to_module.clear()
    saved = []

    for name, module in model.named_modules():
        original_forward = module.forward
        saved.append((module, original_forward))

        # Pre-resolve source location from class definition at patch
        # time.  This is the fallback for eager forward ops — the live
        # traceback inside __torch_dispatch__ is empty under FSDP.
        location = _resolve_module_location(module)

        @functools.wraps(original_forward)
        def wrapped_forward(
            *args, _name=name, _orig=original_forward, _loc=location, **kwargs
        ):
            # Snapshot input grad_fns BEFORE forward runs.  These are
            # the BFS boundary in _record_grad_fn — the walk must not
            # cross into them because they belong to the caller's scope
            # (sibling or parent module), not this module.
            input_grad_fns = _collect_grad_fns(args, kwargs)

            name_token = _current_module_name.set(_name)
            frames_token = _current_stack_frames.set(_loc) if _loc else None
            try:
                result = _orig(*args, **kwargs)
            finally:
                if frames_token is not None:
                    _current_stack_frames.reset(frames_token)
                _current_module_name.reset(name_token)

            # Walk the autograd graph from this module's output to map
            # grad_fn nodes to this module's FQN.  By this point, child
            # modules have already mapped their nodes (they ran during
            # _orig above), so the BFS walks through them without
            # overriding and reaches unmapped nodes that belong to this
            # module (e.g. the mul in FeedForward between w1/w3 and w2).
            _record_grad_fn(result, _name, input_grad_fns)
            return result

        module.forward = wrapped_forward

    return saved


def _unpatch_model(saved: list[tuple[nn.Module, callable]]) -> None:
    """Restore original forward methods saved by _patch_model."""
    for module, original_forward in saved:
        module.forward = original_forward
    _grad_fn_to_module.clear()


def _collect_grad_fns(args, kwargs) -> set:
    """Collect grad_fn objects from input tensors."""
    grad_fns = set()
    for a in torch.utils._pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor) and a.grad_fn is not None:
            grad_fns.add(a.grad_fn)
    return grad_fns


def _record_grad_fn(result, module_name: str, input_grad_fns: set) -> None:
    """Map output grad_fn to module FQN for backward op attribution.

    BFS through the autograd graph from the output grad_fn, mapping
    internal nodes to the module.  Stops at nodes already mapped (child
    modules) and at input grad_fns (sibling/parent module boundary).
    """
    for t in torch.utils._pytree.tree_leaves(result):
        if not isinstance(t, torch.Tensor) or t.grad_fn is None:
            continue
        queue = [t.grad_fn]
        visited = set()
        while queue:
            fn = queue.pop()
            if fn in visited or fn in input_grad_fns:
                continue
            visited.add(fn)
            if fn not in _grad_fn_to_module:
                _grad_fn_to_module[fn] = module_name
            # Walk through child-module nodes (already mapped) to
            # reach unmapped nodes behind them — those belong to
            # this (parent) module.
            for parent, _ in fn.next_functions:
                if parent is not None and parent not in visited:
                    queue.append(parent)


class _CaptureDispatchMode(TorchDispatchMode):
    """Lightweight TorchDispatchMode that clones tensor outputs without wrapping."""

    def __init__(
        self,
        captures: dict[str, CapturedActivation],
        op_counters: dict[str, int],
        min_numel: int,
        op_filter: set[str] | None,
    ):
        super().__init__()
        self.captures = captures
        self.op_counters = op_counters
        self.min_numel = min_numel
        self.op_filter = op_filter
        self._exited = False
        self._in_backward = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        try:
            self._maybe_capture(func, result)
        except Exception:
            pass
        return result

    def _maybe_capture(self, func, result):
        if not _should_capture(func, result, self.min_numel, self.op_filter):
            return

        module_name = _current_module_name.get()

        # For eager backward ops, recover module FQN and source location
        # from the autograd node.  _current_module_name is not set during
        # backward (C++ autograd engine bypasses monkeypatched forwards),
        # but the autograd node maps back to the forward module via
        # _grad_fn_to_module, and detect_anomaly saves forward tracebacks.
        autograd_fwd_trace = None
        autograd_node = torch._C._current_autograd_node()
        if autograd_node is not None:
            if module_name is None:
                module_name = _grad_fn_to_module.get(autograd_node)
            tb = autograd_node.metadata.get("traceback_")
            if tb:
                autograd_fwd_trace = "".join(tb)

        module_name = module_name or "<none>"

        op_name = _get_op_name(func)
        self.op_counters.setdefault(module_name, 0)
        seq = self.op_counters[module_name]
        self.op_counters[module_name] += 1

        key = f"{module_name}/op_{seq}_{op_name}"

        try:
            from torch.distributed.tensor import DTensor

            if isinstance(result, DTensor):
                tensor = result._local_tensor.detach().clone()
            else:
                tensor = result.detach().clone()
        except ImportError:
            tensor = result.detach().clone()

        override_frames = _current_stack_frames.get()
        if override_frames is not None:
            stack_frames = override_frames
        elif autograd_fwd_trace is not None:
            stack_frames = _parse_stack_trace(autograd_fwd_trace)
        else:
            stack = traceback.extract_stack()
            stack_frames = _extract_user_frames(stack)

        # Determine phase:
        # - If _current_phase_override is set, use that.
        # - Default for eager mode: once we see an autograd node, we're in backward
        #   for the rest of this step (sticky flag handles gaps between
        #   node transitions where _current_autograd_node() is None).
        if autograd_node is not None:
            self._in_backward = True

        phase = _current_phase_override.get()
        if phase is None:
            phase = "backward" if self._in_backward else "forward"

        self.captures[key] = CapturedActivation(
            tensor=tensor,
            stack_frames=stack_frames,
            phase=phase,
        )


class ActivationTracer:
    """Zero-source-change activation tracer using monkeypatch + TorchDispatchMode.

    Captures intermediate tensor values during a forward pass, keyed by
    ``module_fqn/op_N_opname``. Works with DTensor and avoids the
    compatibility issues that DebugMode has with DTensor dispatch.

    Example::

        with ActivationTracer(model) as captures:
            output = model(input_batch)

        for key, cap in captures.items():
            print(f"{key}: shape={cap.shape}, loc={cap.stack_frames[-1].short_str()}")

    Args:
        model: The model to trace.
        min_numel: Minimum tensor size to capture (default 1000).
        op_filter: Op name substrings to capture. Default: DEFAULT_COMPUTE_OPS.
            Use None to capture all non-infrastructure ops.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        min_numel: int = 1000,
        op_filter: set[str] | None = DEFAULT_COMPUTE_OPS,
    ):
        self.model = model
        self.min_numel = min_numel
        self.op_filter = op_filter
        self._captures: dict[str, CapturedActivation] = {}
        self._op_counters: dict[str, int] = {}
        self._dispatch_mode: _CaptureDispatchMode | None = None
        self._saved_forwards: list[tuple[nn.Module, callable]] = []

    def __enter__(self) -> dict[str, CapturedActivation]:
        # Skip patching for GraphModules — traced replay uses
        # FQNInterpreter to set context vars from node metadata,
        # and module forwards don't fire during gm(*inputs).
        import torch.fx

        if not isinstance(self.model, torch.fx.GraphModule):
            self._saved_forwards = _patch_model(self.model)
            # Enable detect_anomaly so autograd saves forward tracebacks
            # on backward nodes — used by _maybe_capture to get source
            # locations for eager backward ops.  Not needed for traced
            # mode (node.meta["stack_trace"] provides locations).
            self._anomaly_ctx = torch.autograd.set_detect_anomaly(
                True, check_nan=False
            )
            self._anomaly_ctx.__enter__()
        self._dispatch_mode = _CaptureDispatchMode(
            self._captures,
            self._op_counters,
            self.min_numel,
            self.op_filter,
        )
        self._dispatch_mode.__enter__()
        return self._captures

    def __exit__(self, *args):
        if self._dispatch_mode and not self._dispatch_mode._exited:
            self._dispatch_mode.__exit__(*args)
        if hasattr(self, "_anomaly_ctx"):
            self._anomaly_ctx.__exit__(*args)
        if self._saved_forwards:
            _unpatch_model(self._saved_forwards)
            self._saved_forwards = []


def dump_captures_to_file(
    captures: dict[str, CapturedActivation],
    filepath: str,
) -> None:
    """Write per-op activation statistics to a human-readable log file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"Total captured activations: {len(captures)}\n")
        f.write("=" * 80 + "\n\n")
        for key, cap in captures.items():
            t = cap.tensor
            tf = t.float()
            f.write(f"[{_clean_fqn(key)}]\n")
            f.write(f"  Shape: {t.shape}, Dtype: {t.dtype}\n")
            f.write(f"  L1 norm:  {tf.norm(p=1).item():.6e}\n")
            f.write(f"  L2 norm:  {tf.norm(p=2).item():.6e}\n")
            f.write(f"  Min:      {tf.min().item():.6e}\n")
            f.write(f"  Max:      {tf.max().item():.6e}\n")
            f.write(f"  Mean:     {tf.mean().item():.6e}\n")
            if cap.stack_frames:
                f.write(f"  Location: {cap.stack_frames[-1].short_str()}\n")
            if cap.phase != "forward":
                f.write(f"  Phase: {cap.phase}\n")
            f.write("\n")


class NumericsDebugger:
    """Captures per-op activations on a designated training step.

    Designed to be driven by :class:`Profiler` via its ``step()`` method,
    which is called *after* each training step.  To capture step N the
    debugger arms the tracer after step N-1 completes and dumps after
    step N completes.

    Args:
        enabled: Whether numerics capture is active.
        model: The model whose activations to capture.
        dump_dir: Directory for output log files.
        capture_step: The training step to capture (1-indexed).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        model: nn.Module,
        dump_dir: str,
        capture_step: int = 1,
    ):
        self._enabled = enabled
        self._model = model
        self._dump_dir = dump_dir
        self._capture_step = capture_step
        self._step = 0
        self._captured = False
        self._captures: dict[str, CapturedActivation] | None = None

    def __enter__(self) -> "NumericsDebugger":
        if self._enabled and self._capture_step == 1:
            self._setup()
        return self

    def __exit__(self, *args) -> None:
        self._teardown()

    def step(self) -> None:
        """Called after each training step (same cadence as Profiler.step)."""
        self._step += 1
        if not self._enabled or self._captured:
            return
        if self._step == self._capture_step - 1:
            self._setup()
        elif self._step == self._capture_step:
            self._dump()

    def _setup(self) -> None:
        """Enter ActivationTracer so the next training step is captured."""
        from torchtitan.tools.logging import logger

        logger.info(
            f"Numerics capture: arming for step {self._capture_step}"
        )
        self._captures = {}

        set_numerics_capture_active(True)
        self._tracer = ActivationTracer(self._model)
        self._captures = self._tracer.__enter__()

    def _dump(self) -> None:
        """Dump captures after the capture step completes."""
        from torchtitan.tools.logging import logger

        self._teardown()
        if not self._captures:
            return

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        filepath = os.path.join(self._dump_dir, f"rank_{rank}_activations.log")
        dump_captures_to_file(self._captures, filepath)
        logger.info(f"Dumped {len(self._captures)} activations to {filepath}")
        self._captured = True
        self._captures = None

    def _teardown(self) -> None:
        set_numerics_capture_active(False)
        tracer = getattr(self, "_tracer", None)
        if tracer is not None:
            if tracer._dispatch_mode and not tracer._dispatch_mode._exited:
                tracer.__exit__(None, None, None)
            self._tracer = None
