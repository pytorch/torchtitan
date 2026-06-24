# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerics activation tracer built on top of ``torch.utils._debug_mode.DebugMode``.

Captures per-op activation snapshots during training without modifying
model source code.  Each capture carries the originating module FQN,
source location, phase (forward/backward), and norm-hashes of input +
output tensors (for detecting in-place mutations and bit-exact
divergences).  Works with DTensor, torch.compile, FSDP, and activation
checkpointing.

For each captured op, we track:
    - **Op name**: the ATen operation (e.g. mm, bmm, silu, add)
    - **Module FQN**: which module the op belongs to
      (e.g. ``layers.0.attention.qkv_linear.wq``)
    - **Source location**: stack trace back to the model code that
      triggered the op (e.g. ``common/attention.py:534``)
    - **Phase**: whether the op ran during forward or backward
    - **Output hash / Input hashes**: DebugMode's ``norm_hash_fn``
      applied to result and pre-op inputs.

What gets captured:
    By default, captures outputs of every op that isn't in ``_EXCLUDED_OPS``
    (denylist of infrastructure ops: view, reshape, transpose, clone, copy_,
    detach, collective comms, etc.) and that produces a float tensor with at
    least ``min_numel`` elements (default 1000). This errs on the side of
    capturing too much rather than silently missing newly introduced compute
    ops (e.g. ``_scaled_dot_product_flash_attention``, ``_scaled_mm``).

    Each captured tensor is keyed by ``module_fqn/op_N_opname`` (e.g.
    ``layers.0.attention.qkv_linear.wq/op_0_mm``) and tagged with its
    phase (forward or backward).

Filtering / customization:
    Pass ``op_filter`` to ``DebugModeTracer`` to restrict capture to specific
    ops (matched by substring against the op name). For example, to only
    capture matrix multiplies and softmax::

        DebugModeTracer(model, op_filter={"mm", "bmm", "softmax"})

    Tweak ``min_numel`` to filter small tensors, and edit ``_EXCLUDED_OPS``
    to drop additional infrastructure ops you never want to see.
"""

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
import torch.utils._pytree as pytree
from torch import nn


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
    """A captured activation: pre-computed stats + provenance.

    Stats (Shape, Dtype, L2 norm, Min/Max/Mean) are computed inline at
    capture time so we never hold a tensor clone after the op completes.
    L1 is represented by DebugMode's output hash.  For in-place ops where
    the dispatch returns ``None`` the dict is empty — those entries rely
    on ``input_hashes`` / ``output_hash`` for comparison.
    """

    # Pre-computed stat strings keyed by name (``Shape``, ``L2 norm``,
    # etc.).  Empty for in-place op captures (no result to summarize).
    stats: dict[str, str] = field(default_factory=dict)
    stack_frames: list[_StackFrame] = field(default_factory=list)
    phase: str = "forward"
    # DebugMode norm-hash (basically L1) of the op result.  Comma-
    # separated when the result is a tuple/list of tensors.  Empty
    # string means no hash recorded.
    output_hash: str = ""
    # Comma-separated norm-hashes of input tensors captured *before*
    # the op ran.  For in-place ops (e.g. ``_fused_adam_``) comparing
    # an op's ``input_hashes`` against the previous op's
    # ``input_hashes`` for the same arg surfaces the mutation, even
    # though ``output_hash`` is empty.
    input_hashes: str = ""
    # Producing capture key for each input tensor, aligned positionally
    # with ``input_hashes``.  Resolved from DebugMode's tensor ID
    # tracker: if an input tensor was the output of an earlier captured
    # op, that op's key is stored; otherwise empty.  Semicolon-
    # separated (commas are reserved for the hash list).
    input_producers: str = ""
    # The op's aten name (e.g. ``_fused_adam_``).  Persisted so the
    # comparison tool can recognize in-place ops without re-parsing
    # the key.
    op_name: str = ""


def _compute_stats(tensor: torch.Tensor) -> dict[str, str]:
    """Reduce ``tensor`` to the stat strings written to the log.

    Done inline at capture time so we never have to clone or hold
    onto the tensor itself.  L2 norm and mean are reduced in
    **float64** so they're deterministic across reduction-order drift
    (eager vs traced reorder sums differently — float32 reductions
    diverge in the last digit; float64 has ~9 more bits of precision
    and absorbs it).  L1 norm is intentionally omitted — it's
    identical to ``output_hash`` (which DebugMode computes via
    ``norm_hash_fn`` in float64), so dumping it again would be
    redundant.  Min/Max are exact, so we keep the original dtype.
    """
    t64 = tensor.to(dtype=torch.float64)
    return {
        "Shape": f"{tensor.shape}, Dtype: {tensor.dtype}",
        "L2 norm": f"{t64.norm(p=2).item():.6e}",
        "Min": f"{tensor.float().min().item():.6e}",
        "Max": f"{tensor.float().max().item():.6e}",
        "Mean": f"{t64.mean().item():.6e}",
    }


def _stringify_hash_tree(hash_obj) -> str:
    """Flatten a DebugMode hash pytree into a comma-separated string.

    ``log_tensor_hashes`` stores hashes as a pytree mirroring the
    original args / kwargs / result structure, with floats where
    tensors were and ``None`` elsewhere.  We flatten + drop Nones so
    the log line is human-readable.

    Returns an empty string when nothing hashable was present.
    """
    if hash_obj is None:
        return ""
    leaves = pytree.tree_leaves(hash_obj)
    vals = [v for v in leaves if v is not None]
    if not vals:
        return ""
    return ", ".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in vals)


# Ops to always exclude — infrastructure, not meaningful for numerics parity.
# These are creation, view/reshape, indexing, and collective comm ops that
# either don't perform compute or just rewrap the underlying storage.
_EXCLUDED_OPS = frozenset(
    {
        # Creation
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "lift_fresh",
        "lift_fresh_copy",
        # Copies / dtype conversion
        "copy_",
        "_to_copy",
        "to",
        "clone",
        "contiguous",
        "_pin_memory",
        # View / reshape
        "view",
        "_unsafe_view",
        "view_as",
        "view_as_real",
        "view_as_complex",
        "reshape",
        "_reshape_alias",
        "flatten",
        "unflatten",
        "alias",
        # Permutation
        "permute",
        "transpose",
        "t",
        "unsqueeze",
        "squeeze",
        "expand",
        "expand_as",
        "repeat",
        # Concatenation / splitting
        "cat",
        "stack",
        "split",
        "chunk",
        # Indexing
        "narrow",
        "select",
        "index_select",
        "gather",
        "scatter",
        "slice",
        "as_strided",
        # Autograd plumbing
        "detach",
        # Collective comm (DTensor / FSDP).
        #
        # Async collective ops dispatch and return *before* the
        # destination buffer is filled.
        # "redistribute",
        # "all_gather",
        # "all_gather_into_tensor",
        # "all_gather_into_tensor_out",
        "_pre_bucket_all_gather",
        # "all_reduce",
        # "reduce_scatter",
        # "reduce_scatter_tensor",
        "_pre_bucket_reduce_scatter",
    }
)


def _is_filtered_frame(path: str) -> bool:
    """Filter frames from torch internals and this module.

    Used by ``_parse_stack_trace`` when surfacing source locations
    from ``node.meta["stack_trace"]`` and from DebugMode's
    ``op.fwd_stack_trace`` / ``op.stack_trace`` strings.
    """
    p = path.replace("\\", "/")
    if "/torch/nn/" in p:
        return True
    if "/torch/autograd/" in p:
        return True
    # Dispatcher / compile / runtime plumbing — these frames show up
    # in DebugMode's live traceback under FSDP and activation
    # checkpoint, drowning out the user-code frame.
    if "/torch/_ops.py" in p:
        return True
    if "/torch/_compile.py" in p:
        return True
    if "/torch/_dynamo/" in p:
        return True
    if "/torch/utils/_debug_mode/" in p:
        return True
    if "/torch/utils/checkpoint.py" in p:
        return True
    if "activation_tracer.py" in p:
        return True
    # FQNInterpreter wraps every traced node's execution, so its
    # super().run_node line shows up in the live traceback when a
    # node has no node.meta["stack_trace"].  It's never the
    # location the user wants.
    if "graph_trainer/debug_utils.py" in p:
        return True
    return False


def _get_op_name(func) -> str:
    if hasattr(func, "_schema"):
        return func._schema.name.split("::")[-1]
    return str(func).split(".")[-1].replace(">", "")


def _local_tensor_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor

        return tensor._local_tensor if isinstance(tensor, DTensor) else tensor
    except ImportError:
        return tensor


def _is_captureable_tensor(tensor: torch.Tensor, min_numel: int) -> bool:
    # Skip FakeTensors produced during make_fx tracing.
    if isinstance(tensor, torch._subclasses.FakeTensor):
        return False

    tensor = _local_tensor_if_dtensor(tensor)
    if tensor.numel() < min_numel:
        return False
    return tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}


def _captureable_tensor_leaves(result, min_numel: int) -> list[torch.Tensor]:
    """Return result tensor leaves that are meaningful for numerics capture."""
    tensors: list[torch.Tensor] = []
    for leaf in pytree.tree_leaves(result):
        if isinstance(leaf, torch.Tensor) and _is_captureable_tensor(leaf, min_numel):
            tensors.append(_local_tensor_if_dtensor(leaf))
    return tensors


def _should_capture(func, result, min_numel: int, op_filter: set[str] | None) -> bool:
    """Filter to only capture "interesting" tensor outputs.

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

    Returns:
        True if the result should be captured.
    """
    op_name = _get_op_name(func)
    if op_name in _EXCLUDED_OPS:
        return False
    if op_filter is None:
        return bool(_captureable_tensor_leaves(result, min_numel))
    return any(f in op_name for f in op_filter) and bool(
        _captureable_tensor_leaves(result, min_numel)
    )


# The current module FQN for the op being dispatched.
#
# Set by ``FQNInterpreter.run_node`` during traced graph replay (from
# ``node.meta["custom"]["module_fqn"]``).  Read by
# ``DebugModeTracer.record_hook`` to build the capture key (e.g.
# ``"layers.0.attention.wq/op_0_mm"``).
#
# During eager mode this stays ``None``; the FQN is recovered from
# DebugMode's ``ModTracker`` (for forward ops) or from
# ``_grad_fn_to_module`` (for backward ops).
_current_module_name: ContextVar[str | None] = ContextVar(
    "_current_module_name", default=None
)

# Override for stack frames attached to captured activations.
#
# Set by ``FQNInterpreter`` (from ``node.meta["stack_trace"]``) during
# traced replay.  When set, ``record_hook`` uses these instead of
# DebugMode's live or autograd-forward traceback strings.
_current_stack_frames: ContextVar[list[_StackFrame] | None] = ContextVar(
    "_current_stack_frames", default=None
)

# Phase override for the current op ("forward" or "backward").
#
# Set by ``FQNInterpreter`` from ``node.meta["autograd_backward"]``
# during traced replay.  When set, ``record_hook`` uses this directly
# instead of inferring the phase from ``torch._C._current_autograd_node()``.
_current_phase_override: ContextVar[str | None] = ContextVar(
    "_current_phase_override", default=None
)

# Global flag indicating that ActivationCaptureProfiler has armed capture.
#
# Checked by GraphTrainer._maybe_get_fqn_interpreter() to decide whether to
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


_STACK_TRACE_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*?)(?=\n|$)')


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


# Maps autograd Node (grad_fn) to module FQN. Built during forward
# by the global module forward hooks installed in
# ``DebugModeTracer.__enter__`` (via ``_record_grad_fn``), read during
# backward by ``record_hook`` to give backward ops their originating
# module's FQN.
_grad_fn_to_module: dict[object, str] = {}


def _collect_grad_fns(args, kwargs) -> set:
    """Collect grad_fn objects from input tensors."""
    grad_fns = set()
    for a in pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor) and a.grad_fn is not None:
            grad_fns.add(a.grad_fn)
    return grad_fns


def _record_grad_fn(result, module_name: str, input_grad_fns: set) -> None:
    """Map output grad_fn to module FQN for backward op attribution.

    Walks the autograd graph from the output grad_fn (DFS via an
    explicit stack), mapping internal nodes to the module.  The walk
    stops at input grad_fns (sibling/parent module boundary).  It does
    not overwrite nodes already mapped to child modules, but it still
    walks through them so parent modules can claim unmapped glue ops
    between children.  Traversal order doesn't matter for correctness
    because the "claim only if unclaimed" check is order-independent.
    """
    for t in pytree.tree_leaves(result):
        if not isinstance(t, torch.Tensor) or t.grad_fn is None:
            continue
        stack = [t.grad_fn]
        visited = set()
        while stack:
            fn = stack.pop()
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
                    stack.append(parent)


def dump_captures_to_file(
    captures: dict[str, CapturedActivation],
    filepath: str,
    *,
    skipped_excluded_ops: set[str] | None = None,
) -> None:
    """Write per-op activation statistics to a human-readable log file.

    Args:
        captures: Captured activations to dump.
        filepath: Output path.
        skipped_excluded_ops: Optional set of op names that dispatched
            during the captured step but were dropped because they
            live in ``_EXCLUDED_OPS``.  When provided, written into the
            log header as a comma-separated list so the comparison
            tool can surface them in the HTML.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"Total captured activations: {len(captures)}\n")
        if skipped_excluded_ops is not None:
            ops_str = ", ".join(sorted(skipped_excluded_ops)) or "(none)"
            f.write(f"Excluded ops dispatched: {ops_str}\n")
        f.write("=" * 80 + "\n\n")
        for key, cap in captures.items():
            f.write(f"[{_clean_fqn(key)}]\n")
            if cap.stats:
                # Stats are pre-computed at capture time; the dump is
                # just a write-through.  Order matters for grep-ability,
                # so use a fixed sequence rather than dict iteration.
                for stat in ("Shape", "L1 norm", "L2 norm", "Min", "Max", "Mean"):
                    val = cap.stats.get(stat)
                    if val is not None:
                        f.write(f"  {stat}: {val}\n")
            if cap.output_hash:
                f.write(f"  Output hash: {cap.output_hash}\n")
            if cap.input_hashes:
                f.write(f"  Input hashes: {cap.input_hashes}\n")
            if cap.input_producers:
                f.write(f"  Input producers: {cap.input_producers}\n")
            if cap.stack_frames:
                f.write(f"  Location: {cap.stack_frames[-1].short_str()}\n")
            if cap.phase != "forward":
                f.write(f"  Phase: {cap.phase}\n")
            f.write("\n")


class DebugModeTracer:
    """Captures per-op activations via :class:`torch.utils._debug_mode.DebugMode`.

    Drives the capture by entering a DebugMode with
    ``record_nn_module=True`` (for module FQNs via ``ModTracker``),
    ``record_stack_trace=True`` (for per-op source locations + autograd
    forward tracebacks), ``record_ids=True`` (graph-style tensor IDs
    used for input-producer attribution), and
    ``log_tensor_hashes(hash_inputs=True)`` (norm hashes of inputs and
    outputs, including for in-place ops where the result is ``None``).

    Per-op metadata is stamped by a custom dispatch hook into
    ``op.record``; after ``__exit__`` the captured operator stream is
    walked once to materialize a ``dict[str, CapturedActivation]``
    keyed by ``module_fqn/op_N_opname`` and consumable by
    :mod:`agent_tooling.numerics_debugging.compare_numerics`.

    Args:
        model: The model whose activations to capture.  Used to install
            global ``nn.Module`` forward hooks that populate
            :data:`_grad_fn_to_module` so backward ops can recover the
            originating module's FQN.
        min_numel: Minimum tensor size to capture (default 1000).
        op_filter: Optional set of op-name substrings (allowlist).  None
            captures every op not in ``_EXCLUDED_OPS``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        min_numel: int = 1000,
        op_filter: set[str] | None = None,
    ):
        self.model = model
        self.min_numel = min_numel
        self.op_filter = op_filter
        self._captures: dict[str, CapturedActivation] = {}
        self.skipped_excluded_ops: set[str] = set()
        self._debug_mode = None
        self._hook_ctx = None
        self._hash_ctx = None
        self._materialized = False
        # Global nn.Module forward hook handles, removed on __exit__.
        self._pre_fw_handle = None
        self._post_fw_handle = None

    def __enter__(self) -> dict[str, CapturedActivation]:
        from torch.utils._debug_mode import DebugMode

        self._debug_mode = DebugMode(
            record_realtensor=True,
            record_faketensor=False,
            record_nn_module=True,
            record_stack_trace=True,
            # record_ids populates DebugMode's _tensor_memo with a stable
            # ``$N`` graph ID per tensor.  We share that memo to map
            # input tensors back to the captured op that produced them
            # (used for the producer-link column in the diff HTML).
            record_ids=True,
        )
        # log_tensor_hashes stamps op.log["hash"] (norm hash of the
        # result, basically L1) and op.log["input_hash"] (norm hash of
        # each input tensor *before* the op runs).  Input hashes are
        # what catches in-place mutations — ``_fused_adam_`` returns
        # None so its output is unhashable, but the per-param input
        # hashes are still recorded.
        self._hash_ctx = DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True)

        # Install global forward hooks so backward ops can recover
        # their owning module's FQN via the autograd graph.  DebugMode's
        # ModTracker only fires on Python forward calls, so during
        # backward (driven by the C++ autograd engine) its module stack
        # is empty.  These hooks fire on every nn.Module.__call__:
        # walk the autograd graph from each module's output during the
        # forward post-hook and stash grad_fn -> fqn in _grad_fn_to_module.
        # _maybe_record_module_fqn in record_hook reads it during backward.
        _grad_fn_to_module.clear()
        module_to_fqn = {id(m): name for name, m in self.model.named_modules()}
        pending_input_grad_fns: dict[int, set] = {}

        def pre_fw_hook(module, args):
            mid = id(module)
            if mid not in module_to_fqn:
                return
            pending_input_grad_fns[mid] = _collect_grad_fns(args, {})

        def post_fw_hook(module, args, output):
            mid = id(module)
            fqn = module_to_fqn.get(mid)
            if fqn is None:
                return
            input_grad_fns = pending_input_grad_fns.pop(mid, set())
            _record_grad_fn(output, fqn, input_grad_fns)

        self._pre_fw_handle = nn.modules.module.register_module_forward_pre_hook(
            pre_fw_hook
        )
        self._post_fw_handle = nn.modules.module.register_module_forward_hook(
            post_fw_hook
        )

        # Computes output stats only for ops that pass the filter —
        # cloning every dispatch output (including AC recompute) blows
        # out CUDA memory.  We also stash module FQN, stack frames, and
        # phase per op so the materialize step doesn't need to walk
        # DebugMode annotate entries.
        skipped_excluded_ops = self.skipped_excluded_ops
        min_numel = self.min_numel
        op_filter = self.op_filter
        debug_mode = self._debug_mode

        def record_hook(func, types, args, kwargs, result):
            out = {}
            op_name = _get_op_name(func)
            if op_name in _EXCLUDED_OPS:
                skipped_excluded_ops.add(op_name)
                return out

            # Compute input/output tensor IDs using DebugMode's shared
            # tensor memo — lets _materialize_captures map an input
            # back to its producing op.  Done unconditionally for all
            # non-excluded ops (including in-place ops we won't clone)
            # so the producer map is complete.
            tracker = debug_mode._tensor_memo

            def _id_leaves(obj):
                tree = pytree.tree_map(
                    lambda x: tracker._id(x) if isinstance(x, torch.Tensor) else None,
                    obj,
                )
                return [v for v in pytree.tree_leaves(tree) if v is not None]

            out["input_ids"] = _id_leaves((args, kwargs))
            out["output_ids"] = _id_leaves(result)

            if not _should_capture(func, result, min_numel, op_filter):
                # Still want phase + module FQN + stack frames for the
                # in-place op path in _materialize_captures.  Fall
                # through to the metadata block below.
                pass
            else:
                # Compute stats for the first captureable result tensor.
                # Multi-output ops still get complete DebugMode hash
                # strings for all tensor leaves; the scalar stat columns
                # summarize the primary tensor leaf so the existing log
                # format stays stable.
                primary_tensor = _captureable_tensor_leaves(result, min_numel)[0]
                # Compute stats inline — no clone, no held reference.
                out["stats"] = _compute_stats(primary_tensor)

            # Module FQN priority:
            # 1. _current_module_name ContextVar — set by FQNInterpreter
            #    during traced graph replay (gm bypasses module forwards
            #    so ModTracker is silent).
            # 2. DebugMode's ModTracker stack — populated by Python
            #    module forward hooks in eager mode.  ModTracker roots
            #    FQNs at the model class (``FSDPLlama3Model.layers.0...``)
            #    while named_modules() does not — strip the root segment
            #    so all three sources agree on the same naming.
            # 3. _grad_fn_to_module — populated by our forward hook
            #    (above); recovers FQN for backward ops, where the C++
            #    autograd engine bypasses module forwards so ModTracker
            #    is silent.
            fqn = _current_module_name.get()
            if fqn is None and debug_mode.current_nn_module_stack:
                fqn = debug_mode.current_nn_module_stack[-1]
                root = debug_mode.current_nn_module_stack[0]
                if fqn == root:
                    fqn = None
                elif fqn.startswith(root + "."):
                    fqn = fqn[len(root) + 1 :]
            if fqn is None:
                autograd_node = torch._C._current_autograd_node()
                if autograd_node is not None:
                    fqn = _grad_fn_to_module.get(autograd_node)
            if fqn:
                out["module_fqn"] = _clean_fqn(fqn)

            # Stack frames: ContextVar (from FQNInterpreter, already
            # parsed) takes priority over the live traceback DebugMode
            # captured at dispatch time, which is unhelpful under FSDP /
            # graph replay.
            frames = _current_stack_frames.get()
            if frames is not None:
                out["stack_frames"] = frames

            phase = _current_phase_override.get()
            if phase is None:
                autograd_node = torch._C._current_autograd_node()
                phase = "backward" if autograd_node is not None else "forward"
            out["phase"] = phase
            return out

        self._hook_ctx = DebugMode.dispatch_hooks(record_hook=record_hook)
        self._debug_mode.__enter__()
        self._hash_ctx.__enter__()
        self._hook_ctx.__enter__()
        return self._captures

    def __exit__(self, *args):
        try:
            if self._hook_ctx is not None:
                hook_ctx = self._hook_ctx
                self._hook_ctx = None
                hook_ctx.__exit__(*args)
            if self._hash_ctx is not None:
                hash_ctx = self._hash_ctx
                self._hash_ctx = None
                hash_ctx.__exit__(*args)
            if self._debug_mode is not None:
                self._debug_mode.__exit__(*args)
            if self._debug_mode is not None and not self._materialized:
                self._materialize_captures()
                self._materialized = True
        finally:
            if self._pre_fw_handle is not None:
                self._pre_fw_handle.remove()
                self._pre_fw_handle = None
            if self._post_fw_handle is not None:
                self._post_fw_handle.remove()
                self._post_fw_handle = None
            _grad_fn_to_module.clear()
            # Mark the DebugMode context as closed even if materializing
            # captures raises, so repeated teardown cannot double-exit it.
            self._debug_mode = None

    def _materialize_captures(self) -> None:
        """Build ``module_fqn/op_N_opname`` keyed captures from the
        per-op records stamped by ``record_hook``.

        Most metadata (FQN, stack, phase, output clone) is stamped at
        hook time, so this pass just assigns per-module op indices and
        threads through the tensor-ID producer map used for input-link
        attribution in the diff HTML.
        """
        try:
            from torch.utils._debug_mode import _OpCall
        except ImportError:
            from torch.utils._debug_mode._calls import _OpCall

        op_counters: dict[str, int] = {}
        # Map from tensor ID (from DebugMode's _tensor_memo) to the
        # capture key of the op that produced it.  Populated as we
        # walk operators in dispatch order, so by the time we look up
        # op N's input IDs the map already contains its predecessors'
        # output IDs.  Used to attach ``input_producers`` per capture
        # so the HTML can show "this input came from <op>" on hover.
        producer_map: dict[int, str] = {}

        if self._debug_mode is None:
            return
        for op in self._debug_mode.operators:
            if not isinstance(op, _OpCall):
                continue

            op_name = _get_op_name(op.op)
            record = op.record or {}
            stats = record.get("stats") or {}

            # log_tensor_hashes stamps op.log with input/output norm
            # hashes — populated for every _OpCall DebugMode sees,
            # including in-place ops where result is None.
            log = op.log or {}
            output_hash = _stringify_hash_tree(log.get("hash"))
            input_hashes = _stringify_hash_tree(log.get("input_hash"))

            # Two cases that produce a capture:
            # 1. We have pre-computed stats → normal capture.
            # 2. No stats but we have hashes AND the op name ends with
            #    "_" (in-place convention) and isn't excluded → record
            #    a hash-only capture so optimizer steps / param updates
            #    surface in the diff.
            is_inplace_capture = (
                not stats
                and op_name not in _EXCLUDED_OPS
                and op_name.endswith("_")
                and (output_hash or input_hashes)
            )
            if not stats and not is_inplace_capture:
                continue

            module_name = record.get("module_fqn") or "<none>"
            seq = op_counters.setdefault(module_name, 0)
            op_counters[module_name] = seq + 1
            key = f"{module_name}/op_{seq}_{op_name}"

            stack_frames = record.get("stack_frames")
            if not stack_frames:
                # Fall back to DebugMode's own traceback strings.
                # fwd_stack_trace is the autograd-saved forward stack
                # (useful on backward ops); stack_trace is the live
                # dispatch-time stack (poor under FSDP / replay).
                trace_str = getattr(op, "fwd_stack_trace", None) or getattr(
                    op, "stack_trace", None
                )
                stack_frames = _parse_stack_trace(trace_str) if trace_str else []

            # Resolve each input tensor ID to the capture key that
            # produced it (if any).  Positional: empty string means
            # "produced by an uncaptured op or external input
            # (parameter, dataloader)".
            input_ids = record.get("input_ids") or []
            input_producers = ";".join(producer_map.get(tid, "") for tid in input_ids)

            self._captures[key] = CapturedActivation(
                stats=stats,
                stack_frames=stack_frames,
                phase=record.get("phase", "forward"),
                output_hash=output_hash,
                input_hashes=input_hashes,
                input_producers=input_producers,
                op_name=op_name,
            )

            # Register this op as the producer of its output tensors so
            # downstream consumers can find it.
            for tid in record.get("output_ids") or []:
                producer_map[tid] = key


class ActivationCaptureProfiler:
    """Profiler-driven companion that captures per-op activations on a designated step.

    Designed to be driven by :class:`Profiler` via its ``step()`` method,
    which is called *after* each training step.  To capture step N this
    profiler arms ``DebugModeTracer`` after step N-1 completes and dumps
    after step N completes.

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
        self._tracer: DebugModeTracer | None = None
        self._captures: dict[str, CapturedActivation] | None = None

    def __enter__(self) -> "ActivationCaptureProfiler":
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
        """Enter DebugModeTracer so the next training step is captured."""
        from torchtitan.tools.logging import logger

        logger.info(f"Numerics capture: arming for step {self._capture_step}")
        set_numerics_capture_active(True)
        self._tracer = DebugModeTracer(self._model)
        self._captures = self._tracer.__enter__()

    def _dump(self) -> None:
        """Dump captures after the capture step completes."""
        from torchtitan.tools.logging import logger

        # _teardown() exits the tracer.  DebugModeTracer populates
        # skipped_excluded_ops inside __exit__ (when operators are
        # walked), so we snapshot it *after* teardown.
        tracer = self._tracer
        self._teardown()
        skipped = set(tracer.skipped_excluded_ops) if tracer is not None else set()
        if not self._captures:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = os.path.join(self._dump_dir, f"rank_{rank}_activations.log")
        dump_captures_to_file(self._captures, filepath, skipped_excluded_ops=skipped)
        logger.info(f"Dumped {len(self._captures)} activations to {filepath}")
        self._captured = True
        self._captures = None

    def _teardown(self) -> None:
        set_numerics_capture_active(False)
        tracer = self._tracer
        self._tracer = None
        if tracer is None:
            return
        if tracer._debug_mode is not None:
            tracer.__exit__(None, None, None)
