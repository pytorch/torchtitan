# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CUDAGraph pass for the graph trainer.

This module provides a cudagraph pass that can be applied to graph modules
during compilation.
"""

import gzip
import json
import operator
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor.cudagraph_trees import _use_cuda_memory_pool_manager
from torch._library.opaque_object import is_opaque_value
from torch.fx.passes.split_module import split_module
from torch.utils._ordered_set import OrderedSet

from torchtitan.config.function import Function
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.debug_utils import tlparse_log_graph_pass
from torchtitan.tools.logging import logger


class _CUDAGraphManager:
    """A manager to hold a shared graph pool, stream, and wrapper registry."""

    def __init__(self) -> None:
        self._initialized = False
        self._cudagraph_wrappers: list["CUDAGraphWrapper"] = []
        self._teardown_called = False
        # toolsId (graph_id << 32 | node_id) -> list of annotation dicts
        # (e.g. [{"module_fqn": "layers.0.attention.wq"}]).
        self.all_annotations: dict[int, list] = {}
        self.enable_annotations: bool = False

    def maybe_initialize(self) -> None:
        if self._initialized:
            return

        self._initialized = True

        # create a global cudagraph memory pool to allow memory reuse across cudagraphs.
        self.graph_pool = torch.cuda.graph_pool_handle()

        # create a global cuda stream for graph capture. we need to use a single stream
        # for all allocations to the memory pool, otherwise the allocations to separate
        # streams will not be used.
        self.stream = torch.cuda.Stream()

        # use a dummy graph to keep the global graph pool alive
        self._dummy_graph = torch.cuda.CUDAGraph()
        with (
            # suppress an empty cudagraph warning, since we intentionally create
            # an empty cudagraph here
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self._dummy_graph,
                pool=self.graph_pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass

    def register_wrapper(self, wrapper: "CUDAGraphWrapper") -> None:
        assert not self._teardown_called, "Cannot register new cudagraph after teardown"
        self._cudagraph_wrappers.append(wrapper)

    def teardown(self) -> None:
        """Destroy all cudagraphs and release the cudagraph memory pool.

        Note [explicit cudagraph teardown]
        cudagraph holds reference to nccl which prevents destroy process
        group. so we need to explicitly delete cudagraph which is held
        in _CUDAGraphManager and CUDAGraphWrapper. If cudagraph is not
        used, this is a no-op.
        """
        if not self._initialized:
            return
        if self._teardown_called:
            logger.warning("cudagraph manager teardown called twice")
            return

        for wrapper in self._cudagraph_wrappers:
            wrapper.teardown()
        self._cudagraph_wrappers.clear()

        self._dummy_graph = None
        self.stream = None
        self.graph_pool = None
        self._teardown_called = True


_cg_manager = _CUDAGraphManager()


def cudagraph_teardown() -> None:
    """Destroy all cudagraphs and release the cudagraph memory pool.
    See Note [explicit cudagraph teardown] for more details.
    """
    _cg_manager.teardown()


def get_cudagraph_annotations() -> dict[int, list]:
    """Return all kernel annotations accumulated across CUDA graph captures."""
    return _cg_manager.all_annotations


def enable_cudagraph_annotations() -> None:
    """Enable kernel annotation capture on subsequent CUDA graph recordings."""
    _cg_manager.enable_annotations = True


def cudagraph_annotate_trace_post_processor() -> Function.Config:
    """Return a ``Function.Config`` that merges captured CUDA graph kernel
    annotations into a profiler trace file.

    Attach this to ``Profiler.Config.trace_post_processor`` so that exported
    profiler traces automatically carry ``module_fqn`` fields on graphed kernel
    events.
    """
    return Function.Config(fn=_cudagraph_annotate_trace_file)


def _cudagraph_annotate_trace_file(trace_path: str) -> None:
    """Post-process a profiler trace with CUDA graph kernel annotations."""
    annotations = _cg_manager.all_annotations
    if not annotations:
        return

    try:
        from torch.cuda._annotate_cuda_graph_trace import (  # pyrefly: ignore[missing-import]
            annotate_trace,
        )
    except ImportError:
        logger.warning(
            "torch.cuda._annotate_cuda_graph_trace not available. "
            "Upgrade PyTorch to enable trace CUDA graph kernel annotation."
        )
        return

    # Profiler.export_chrome_trace gzip-compresses when the path ends in
    # ".gz" (the default PROFILE_FILE since #3483), so read/write the trace
    # through gzip for those paths and fall back to plain text otherwise.
    open_trace = gzip.open if trace_path.endswith(".gz") else open

    with open_trace(trace_path, "rt") as f:
        trace = json.load(f)

    count = annotate_trace(trace, annotations)
    if count > 0:
        with open_trace(trace_path, "wt") as f:
            json.dump(trace, f)
        logger.info(f"Annotated {count} CUDAGraph kernel event(s) in profiler trace")


class CUDAGraphWrapper:
    """Wraps a callable with cudagraph. It warms up the callable, records cudagraph,
    and replays cudagraph during runtime. It also handles static input tensors, which
    are tensors whose tensor addresses do not change across runs.

    Args:
        runnable: The callable to wrap with CUDA graph. This can be a
            torch.fx.GraphModule when used in an FX graph pass, or any
            callable when used in PyTorch eager mode.
        example_inputs: A list of example inputs to the callable.
        static_input_indices: A tuple of indices identifying static input
            tensors. Static inputs are tensors whose memory addresses remain
            constant across invocations. Common examples include model weights,
            buffers, and outputs from previously wrapped CUDA graph functions.
        should_check_address: Whether to verify static input tensor addresses
            at runtime. This should only be enabled for debugging purposes.
    """

    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int] | None = None,
        should_check_address: bool = False,
        tensor_input_indices: list[int] | None = None,
    ):
        _cg_manager.maybe_initialize()
        _cg_manager.register_wrapper(self)

        self._runnable = runnable
        self._static_input_indices = OrderedSet(
            static_input_indices if static_input_indices is not None else []
        )
        if tensor_input_indices is not None:
            self._input_indices_to_copy = [
                i for i in tensor_input_indices if i not in self._static_input_indices
            ]
        else:
            self._input_indices_to_copy = [
                i
                for i, inp in enumerate(example_inputs)
                if isinstance(inp, torch.Tensor) and i not in self._static_input_indices
            ]
        self._cudagraph: torch.cuda.CUDAGraph | None = None
        self._has_warmup = False

        self._args = None
        self._output = None

        # (debug only) whether check static input tensor addresses during runtime
        self._should_check_address = should_check_address

        self._gm = runnable if isinstance(runnable, torch.fx.GraphModule) else None

    def print_readable(self, *args, **kwargs):
        """Delegate to the inner GraphModule's print_readable."""
        assert self._gm is not None, "print_readable requires a GraphModule runnable"
        return self._gm.print_readable(*args, **kwargs)

    def _copy_non_static_inputs(self, *args):
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])

    def _validate_inputs(self, inputs) -> None:
        """Validate that all inputs are of supported types.

        Opaque inputs (e.g. DeviceMesh from SimpleFSDP/DTensor) are
        inherently static and already excluded from copying (only
        tensors appear in ``_input_indices_to_copy``), so no special
        handling is needed beyond accepting them here.
        """
        for i, inp in enumerate(inputs):
            if isinstance(inp, (torch.Tensor, int, float, torch._C.Generator)):
                continue
            if is_opaque_value(inp):
                continue
            raise ValueError(
                "args must be tensor, integer (for dynamic shapes), "
                "float (for scalar constants), "
                "Generator (for random number generator), "
                "or opaque object, "
                f"but found {type(inp)} with value {inp!r} at index {i}"
            )

    def _check_static_inputs_address(self) -> None:
        for i in self._static_input_indices:
            actual = self._args[i].data_ptr()
            expected = self._input_addresses[i]
            assert expected == actual, (
                "Expected the same static tensor address but found "
                f"{expected} != {actual}"
            )

    def __call__(self, *args):
        if not self._has_warmup:
            self._has_warmup = True
            device = torch.cuda.current_device()

            # warmup in cudagraph memory pool to avoid fragmentation
            # across eager memory pool and cudagraph memory pool.
            with _use_cuda_memory_pool_manager(
                device, _cg_manager.graph_pool, _cg_manager.stream
            ):
                out = self._runnable(*args)
            return out

        if self._cudagraph is None:
            self._validate_inputs(args)
            self._args = args
            self._input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args
            ]

            self._cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(
                self._cudagraph,
                pool=_cg_manager.graph_pool,
                stream=_cg_manager.stream,
                enable_annotations=_cg_manager.enable_annotations,
            ):
                # `output` is managed by pytorch's cudagraph pool
                self._output = self._runnable(*args)

            if _cg_manager.enable_annotations:
                from torch.cuda._graph_annotations import get_kernel_annotations

                _cg_manager.all_annotations.update(get_kernel_annotations())

        if self._should_check_address:
            self._check_static_inputs_address()

        self._copy_non_static_inputs(*args)
        self._cudagraph.replay()
        return self._output

    def teardown(self) -> None:
        """Destroy cudagraph and release references.
        See Note [explicit cudagraph teardown] for more details.
        """
        self._cudagraph = None
        self._args = None
        self._output = None


def _has_dynamic_shape(val: Any) -> bool:
    """True if ``val`` is (or contains) a tensor with a symbolic (data-dependent)
    shape — i.e. any dimension is a ``torch.SymInt`` rather than a concrete int."""
    if isinstance(val, torch.Tensor):
        return any(isinstance(s, torch.SymInt) for s in val.shape)
    if isinstance(val, (list, tuple)):
        return any(_has_dynamic_shape(v) for v in val)
    return False


def _iter_tensors(val: Any) -> list[torch.Tensor]:
    """Flatten ``val`` (tensor / list / tuple) to the tensors it contains."""
    if isinstance(val, torch.Tensor):
        return [val]
    if isinstance(val, (list, tuple)):
        return [t for v in val for t in _iter_tensors(v)]
    return []


def is_cudagraphable(node: torch.fx.Node) -> bool:
    """Whether ``node`` can be captured by a CUDA graph.

    Per-node predicate for the partitioner (:func:`cudagraph_pass`) and the
    build-time gate (:func:`is_full_cudagraphable`).

    flex_attention HOPs count as cudagraphable: regional_inductor compiles them to
    Triton kernels before cudagraph, so this never sees a flex HOP at capture time.
    """
    if node.op != "call_function":
        return True

    # getitem only indexes a multi-output op's result, so it inherits its parent's
    # cudagraph-ability rather than being judged on its own tensors (e.g. a getitem
    # of an eager dynamic-shape/CPU op must also be eager).
    if node.target is operator.getitem:
        parent = node.args[0]
        return not isinstance(parent, torch.fx.Node) or is_cudagraphable(parent)

    # Cross-device copy from/to unpinned CPU memory: cudagraph requires the CPU
    # side to be pinned for the async H2D/D2H copy.
    if node.target in (
        torch.ops.aten.copy_.default,
        torch.ops.aten._to_copy.default,
    ):
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            for inp in node.all_input_nodes:
                inp_val = inp.meta.get("val")
                if (
                    isinstance(inp_val, torch.Tensor)
                    and inp_val.device.type != val.device.type
                ):
                    cpu_val = val if val.device.type == "cpu" else inp_val
                    if not cpu_val.is_pinned():
                        return False

    # aten._grouped_mm may perform internal CPU<->CUDA copies not visible in FX
    # metadata; resolved on sm_100+.
    if node.target == torch.ops.aten._grouped_mm.default:
        if torch.cuda.get_device_capability() < (10, 0):
            return False

    # .item()/.tolist() need a device-to-host sync a cudagraph replay can't redo.
    if node.target == torch.ops.aten._local_scalar_dense.default:
        return False

    # Op with a dynamic (data-dependent / unbacked-SymInt) input or output shape.
    if _has_dynamic_shape(node.meta.get("val")) or any(
        _has_dynamic_shape(inp.meta.get("val")) for inp in node.all_input_nodes
    ):
        return False

    # Pure-CPU op: every tensor in its inputs and output lives on CPU. A CUDA graph
    # only captures CUDA kernels -- a CPU op would run on the host at capture time
    # and replay stale (e.g. the .tolist() unbind/getitem on EP token-routing split
    # sizes). Run it eager.
    tensors = _iter_tensors(node.meta.get("val"))
    for inp in node.all_input_nodes:
        tensors += _iter_tensors(inp.meta.get("val"))
    if tensors and all(t.device.type == "cpu" for t in tensors):
        return False

    return True


def is_full_cudagraphable(gm: torch.fx.GraphModule) -> bool:
    """True if every node is cudagraphable (:func:`is_cudagraphable`), i.e. the
    graph can be captured as one full CUDA graph. Used to resolve
    ``cudagraph_mode='auto'`` at pipeline-build time."""
    return all(is_cudagraphable(node) for node in gm.graph.nodes)


def get_static_input_indices(gm: torch.fx.GraphModule, is_forward: bool) -> list[int]:
    """
    Get indices of gm inputs that are static input tensors whose tensor addresses do not
    change across runs. Example of static input tensors include weights, buffers, and
    outputs of previous cudagraph wrapped functions.
    """
    from torch._inductor.utils import count_tangents

    static_input_indices = []
    if (
        is_forward
        and (tracing_context := torch._guards.TracingContext.try_get())
        and hasattr(tracing_context, "fw_metadata")
    ):
        # for forward, we rely on graph capture (i.e., dynamo or export) to provide
        # the correct static input indices stored in tracing context. Typical examples
        # include weights and buffers.
        static_input_indices = tracing_context.fw_metadata.static_input_indices

    elif not is_forward:
        # for backward, we identify saved tensors as static inputs, since saved tensors
        # are outputs of cudagraph-wrapped forward run. In PT2-generated backward gm,
        # saved tensors are always the leading args. So we can get the number of saved
        # tensors and generate static input indices.
        fixed = count_tangents(gm)
        static_input_indices = list(range(fixed))

    return static_input_indices


def insert_kernel_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Insert mark_kernels() calls at module boundaries in the FX graph.

    Reads ``node.meta["custom"]["module_fqn"]`` (set via
    ``annotate_module_fqns``) and inserts enter/exit calls so that
    CUDA graph capture records the annotations.

    Requires ``cuda-python`` package and CUDA toolkit/driver >= 13.1
    (or cuda-compat >= 13.1).  Returns the graph unchanged when unavailable.

    Also enables annotation capture on :class:`CUDAGraphWrapper` so that
    ``enable_annotations=True`` is passed to ``torch.cuda.graph()``.

    Alternative approaches:

    1. **fx.Interpreter**: During cudagraph capture, run the graph via an
       ``fx.Interpreter`` subclass that reads ``module_fqn`` metadata and
       calls ``mark_kernels`` enter/exit around each node — avoids mutating
       the graph.
    2. **Custom CodeGen**: Use a custom ``torch.fx.graph.CodeGen`` to emit
       enter/exit lines (or ``with`` blocks) directly in the generated
       Python code.

    The current graph-pass approach is the least invasive.
    """
    from torch.cuda._graph_annotations import _is_tools_id_unavailable

    def _enter(annotation: dict) -> object:
        from torch.cuda._graph_annotations import mark_kernels

        ctx = mark_kernels(annotation)
        ctx.__enter__()
        return ctx

    def _exit(ctx: object) -> None:
        ctx.__exit__(None, None, None)  # type: ignore[union-attr]

    if _is_tools_id_unavailable():
        return gm

    enable_cudagraph_annotations()

    graph = gm.graph
    current_fqn: str | None = None
    current_ctx_node = None

    for node in list(graph.nodes):
        fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)

        if fqn != current_fqn:
            # Close previous scope
            if current_ctx_node is not None:
                with graph.inserting_before(node):
                    exit_node = graph.call_function(_exit, (current_ctx_node,))
                    exit_node.meta["custom"] = {}
                current_ctx_node = None

            # Open new scope
            if fqn is not None:
                with graph.inserting_before(node):
                    enter_node = graph.call_function(
                        _enter,
                        ({_MODULE_FQN: fqn},),
                    )
                    enter_node.meta["custom"] = {}
                current_ctx_node = enter_node

            current_fqn = fqn

    # Close any trailing scope (before output/return)
    if current_ctx_node is not None:
        output_nodes = [n for n in graph.nodes if n.op == "output"]
        if output_nodes:
            with graph.inserting_before(output_nodes[0]):
                exit_node = graph.call_function(_exit, (current_ctx_node,))
                exit_node.meta["custom"] = {}

    graph.lint()
    gm.recompile()
    return gm


def full_cudagraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool = True,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
) -> torch.fx.GraphModule:
    """Capture the entire joint graph as one CUDA graph — the single-subgraph case
    of :func:`cudagraph_pass`.

    Wraps ``gm.forward`` with a :class:`CUDAGraphWrapper`; capture happens lazily
    at runtime (warm up on call 1, record on call 2, replay after).

    Args:
        gm: The graph module to wrap.
        example_inputs: Example inputs for warmup/recording.
        is_forward: Whether to infer static-input indices as for a forward graph
            (used only when ``static_input_indices`` is not provided).
        static_input_indices: Explicit list of input indices with stable tensor
            addresses (params/buffers).
        tensor_input_indices: Indices of graph inputs that are tensors (vs opaque
            values like DeviceMesh). When not provided, inferred from
            ``example_inputs``.
    """
    if static_input_indices is None:
        static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(
        gm.forward,
        example_inputs,
        static_input_indices,
        tensor_input_indices=tensor_input_indices,
    )
    logger.info("Applied full cudagraph pass (single capturable subgraph).")
    return gm


def _node_module_fqn(node: torch.fx.Node) -> str | None:
    """The ``module_fqn`` annotation on ``node`` (set by ``annotate_module_fqns``),
    or None if unannotated."""
    return (node.meta.get("custom") or {}).get(_MODULE_FQN)


# DSv3 cudagraph carve-out. The routed experts (``*.moe.experts``) own the
# cudagraph-incompatible ops (sm_90 ``_grouped_mm``, the EP all-to-all dispatch host
# sync), so that subtree runs eager; everything else -- including the rest of the MoE
# block (router, shared_experts) and attention/norms/embedding/lm_head -- is captured.
# Hardcoded by ``module_fqn`` -- this is the DSv3-specific partitioner. The boundaries
# (``(?:^|\.)`` / ``(?:\.|$)``) match ``moe.experts`` as whole dotted components, so
# siblings like ``moe.shared_experts`` do not match.
_CARVE_OUT_FQN_RE = re.compile(r"(?:^|\.)moe\.experts(?:\.|$)")


def _is_carved_out(fqn: str | None) -> bool:
    """Return True if ``fqn`` lives inside the carved-out (eager) subtree -- the
    routed experts ``*.moe.experts`` (e.g. ``layers.3.moe.experts`` or
    ``layers.3.moe.experts.w1``). The router and shared_experts siblings under
    ``moe`` are *not* carved out -- they capture fine."""
    return bool(fqn) and _CARVE_OUT_FQN_RE.search(fqn) is not None


# FSDP collectives whose bucket chains bucketing software-pipelines (issues the
# all-gather ~one transformer block before its ``wait_tensor`` to overlap with prior
# compute). In a *piecewise* capture these chains must run EAGER: a functional
# collective registers async work at the collective and consumes it at ``wait_tensor``,
# but the prefetch leaves the collective and its wait ~one segment apart, so they land in
# different CUDA-graph segments -- and a collective recorded in one graph is an invalid
# work handle when waited in another segment (``cudaErrorInvalidValue``). Keeping the
# whole chain (collective + wait + bucket prep) eager co-locates them. The two
# directions -- all-gather (forward weight prefetch) and reduce-scatter (backward grad) --
# are handled identically; the split constants exist so each can be exercised in
# isolation. Matched by overload packet. See :func:`_collective_chain_nodes`.
_ALL_GATHER_PACKETS = frozenset(
    {
        torch.ops._c10d_functional.all_gather_into_tensor,
        torch.ops._c10d_functional.all_gather_into_tensor_out,
    }
)
_REDUCE_SCATTER_PACKETS = frozenset(
    {
        torch.ops._c10d_functional.reduce_scatter_tensor,
        torch.ops._c10d_functional.reduce_scatter_tensor_out,
    }
)
_COLLECTIVE_PACKETS = _ALL_GATHER_PACKETS | _REDUCE_SCATTER_PACKETS
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default

# Bucket buffer-construction ops (by op base name): the data-movement ops that build a
# collective's contiguous input buffer -- the ``_pre_bucket_*`` custom ops plus the
# ``cat``/``copy``/``slice``/``empty`` they expand into. The collective-chain walk absorbs
# ONLY these, so the *compute* that produces the values being bucketed (e.g. the
# weight-grad ``mm``/``_grouped_mm`` feeding a reduce-scatter) is NOT dragged into the
# chain -- it stays capturable. A buffer op left out of this set merely stays in its own
# segment (a cheap data-movement node captured separately), which is harmless; a compute
# op wrongly added would force real compute eager, so keep this to pure buffer plumbing.
#
# NOTE: the *output-side* unpacking (``split_with_sizes`` -> ``getitem`` -> ``view`` that
# unflattens the gathered bucket) is deliberately NOT absorbed. Leaving it captured copies
# the gathered bucket into one cudagraph input buffer; forcing it eager instead copies each
# unflattened per-parameter weight into its captured consumer's buffer -- many pinned copy
# buffers vs one, which measured ~2.6x the cudagraph-pool footprint and OOM'd.
_BUCKET_BUFFER_OP_NAMES = frozenset(
    {
        "_pre_bucket_all_gather",
        "_pre_bucket_reduce_scatter",
        "slice",
        "cat",
        "copy",
        "copy_",
        "empty",
        "new_empty",
    }
)


def _is_bucket_buffer_op(node: torch.fx.Node) -> bool:
    """True if ``node`` is a bucket buffer-construction op (see
    :data:`_BUCKET_BUFFER_OP_NAMES`), matched by overload-packet base name."""
    packet = getattr(node.target, "overloadpacket", None)
    if packet is None:
        return False
    return str(packet).split(".")[-1] in _BUCKET_BUFFER_OP_NAMES


def _collective_chain_nodes(
    gm: torch.fx.GraphModule, packets: frozenset
) -> set[torch.fx.Node]:
    """Return the nodes forming the FSDP collective chains for ``packets``.

    A chain is a collective (matched by overload packet), its ``wait_tensor``, and the
    bucket *buffer-construction* DAG that exists solely to feed it -- e.g. the all-gather
    prefetch ``_pre_bucket_all_gather`` -> ``slice`` -> ``all_gather_into_tensor[_out]``
    -> ``wait_tensor`` pattern (bucketing custom-op mode). The caller forces the whole
    chain eager in piecewise mode so the collective and its ``wait_tensor`` stay
    co-located (see :data:`_COLLECTIVE_PACKETS` for why capturing a prefetch-separated
    pair breaks).

    The backward walk absorbs an input iff it is a buffer-construction op
    (:func:`_is_bucket_buffer_op`) AND all of its users are already in the chain. The
    buffer-op restriction is load-bearing: the *values* being bucketed (the weight-grad
    ``mm``/``_grouped_mm`` feeding a reduce-scatter) also satisfy "all users in chain",
    so without it the walk would drag that whole capturable backward-compute subtree into
    the (eager) chain -- exactly the over-absorption that strands real compute eager.
    """
    chain: set[torch.fx.Node] = set()
    worklist: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if getattr(node.target, "overloadpacket", None) in packets:
            chain.add(node)
            worklist.append(node)
            for user in node.users:
                if user.op == "call_function" and user.target is _WAIT_TENSOR:
                    chain.add(user)
    while worklist:
        node = worklist.pop()
        for inp in node.all_input_nodes:
            if inp in chain or inp.op != "call_function":
                continue
            if _is_bucket_buffer_op(inp) and all(
                user in chain for user in inp.users
            ):
                chain.add(inp)
                worklist.append(inp)
    return chain


def _producer_call_node(arg: Any) -> torch.fx.Node | None:
    """Return the ``call_module`` node that produced ``arg`` in the top-level
    split graph, or None. Handles both single-output submodules (the arg is the
    call node directly) and multi-output submodules (the arg is a ``getitem`` of
    the call node)."""
    if not isinstance(arg, torch.fx.Node):
        return None
    if arg.op == "call_module":
        return arg
    if arg.op == "call_function" and arg.target is operator.getitem:
        parent = arg.args[0]
        if isinstance(parent, torch.fx.Node) and parent.op == "call_module":
            return parent
    return None


def _compute_partition(
    gm: torch.fx.GraphModule, min_capture_size: int = 1
) -> tuple[dict[torch.fx.Node, int], dict[int, bool]]:
    """Partition the graph into contiguous eager/capturable runs by ``module_fqn``.

    Carve out the routed experts (``*.moe.experts``, see :func:`_is_carved_out`) to run
    eager -- they own the cudagraph-incompatible ops (sm_90 ``_grouped_mm``, EP
    token-routing host syncs) -- and capture everything else (attention, norms, the rest
    of the MoE block, embedding, lm_head). When partitioning piecewise, FSDP collective
    chains are additionally forced eager (see :func:`_collective_chain_nodes` and
    :data:`_COLLECTIVE_PACKETS`): bucketing software-pipelines the collective ~one block
    before its ``wait_tensor``, so capturing them would split collective from wait across
    segments and invalidate the wait. Contiguous same-kind runs share one monotonic
    subgraph id (split_module's grad/autocast requirement).

    A capturable run shorter than ``min_capture_size`` is demoted to eager (a private
    pool + per-step copy-in isn't worth a handful of ops); since such runs are always
    flanked by eager runs, this just coalesces them (numerically neutral).

    Returns ``(node_to_subgraph_id, subgraph_is_eager)``, the latter mapping each
    subgraph id to whether it runs eager. Also stamps observability tags
    ``node.meta["custom"]["cudagraphable"]`` and ``["partition_id"]`` on each node.
    """
    # A call_function node is eager iff its module lives in the carved-out block;
    # get_attr/etc. are neutral and ride whichever run they fall in.
    real_nodes = [n for n in gm.graph.nodes if n.op not in ("placeholder", "output")]
    node_is_eager: dict[torch.fx.Node, bool] = {
        node: node.op == "call_function" and _is_carved_out(_node_module_fqn(node))
        for node in real_nodes
    }

    # Piecewise only (something is carved out): force the FSDP collective chains eager so
    # each collective and its ``wait_tensor`` stay co-located on the eager side. Bucketing
    # software-pipelines them ~one transformer block apart, so capturing them would split
    # collective from wait across CUDA-graph segments -- an invalid work handle at the wait
    # (``cudaErrorInvalidValue``). See :data:`_COLLECTIVE_PACKETS`. Skipped for a full
    # capture (nothing carved out): there are no segment boundaries, so the collectives and
    # their waits live in the one graph and capture fine -- forcing them eager would
    # needlessly fragment that path.
    if any(node_is_eager.values()):
        for node in _collective_chain_nodes(gm, _COLLECTIVE_PACKETS):
            node_is_eager[node] = True

    # Linear scan over the contiguous eager/capturable runs: demote any capturable run
    # shorter than ``min_capture_size`` to eager -- a lone collective launch or a handful
    # of ops stranded in the eager experts region isn't worth a separate CUDA graph +
    # per-step copy-in, and folding it coalesces the surrounding eager runs (numerically
    # neutral, since capturable runs are always flanked by eager ones). Skip entirely when
    # nothing is carved out: the whole graph is one capturable run for a full capture, and
    # demoting it would wrongly force it eager (failing require_full).
    if min_capture_size > 1 and any(node_is_eager.values()):
        i = 0
        n = len(real_nodes)
        while i < n:
            j = i
            while (
                j < n and node_is_eager[real_nodes[j]] == node_is_eager[real_nodes[i]]
            ):
                j += 1
            if not node_is_eager[real_nodes[i]] and (j - i) < min_capture_size:
                for k in range(i, j):
                    node_is_eager[real_nodes[k]] = True
            i = j

    # Assign contiguous-run subgraph ids from the (possibly demoted) eager status,
    # recording each subgraph's eager/capturable kind as we go.
    node_to_subgraph_id: dict[torch.fx.Node, int] = {}
    subgraph_is_eager: dict[int, bool] = {}
    subgraph_id = 0
    prev_eager: bool | None = None
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            node_to_subgraph_id[node] = subgraph_id
            continue
        eager = node_is_eager[node]
        if prev_eager is None or eager != prev_eager:
            subgraph_id += 1
            prev_eager = eager
        node_to_subgraph_id[node] = subgraph_id
        subgraph_is_eager[subgraph_id] = eager
        # Observability tags (carried onto submodule nodes by split_module):
        # whether the node is captured, and the partition it lands in ("cg_<id>" for a
        # captured cudagraph region, "eager_<id>" for an eager region). Copy ``custom``
        # into a fresh dict before writing: bucketing creates the FSDP collective-chain
        # nodes sharing their origin experts node's ``meta["custom"]`` object, so an
        # in-place write here would scramble both nodes' tags (the later write wins) --
        # making the logged ``partition_id`` non-monotonic even though
        # ``node_to_subgraph_id`` is correct. A private copy keeps each node's tag equal
        # to its own subgraph id.
        custom = dict(node.meta.get("custom") or {})
        custom["cudagraphable"] = not eager
        custom["partition_id"] = (
            f"eager_{subgraph_id}" if eager else f"cg_{subgraph_id}"
        )
        node.meta["custom"] = custom
    return node_to_subgraph_id, subgraph_is_eager


def _colocate_getitems(
    gm: torch.fx.GraphModule, node_to_subgraph_id: dict[torch.fx.Node, int]
) -> None:
    """Pull every getitem into its (multi-output) parent's subgraph.

    A getitem indexes a node whose value is a tuple/list; that value must not cross a
    submodule boundary because :class:`CUDAGraphWrapper` rejects list/tuple inputs.
    For each getitem not already in its parent's subgraph, move it to immediately
    after its parent (keeping the parent's run contiguous) and assign it the parent's
    subgraph id. Topologically safe: a getitem depends only on its parent, so moving
    it earlier keeps all of its users after it. Mutates ``gm.graph`` and
    ``node_to_subgraph_id`` in place.
    """
    moved = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not operator.getitem:
            continue
        parent = node.args[0]
        if not isinstance(parent, torch.fx.Node) or parent not in node_to_subgraph_id:
            continue
        if node_to_subgraph_id.get(node) == node_to_subgraph_id[parent]:
            continue
        parent.append(node)  # move adjacent -> parent's run stays contiguous
        node_to_subgraph_id[node] = node_to_subgraph_id[parent]
        # Private copy before writing -- the getitem may share its ``meta["custom"]``
        # object with another node (see :func:`_compute_partition`), so an in-place
        # write would scramble that node's tag too.
        custom = dict(node.meta.get("custom") or {})
        parent_custom = parent.meta.get("custom", {})
        custom["cudagraphable"] = parent_custom.get(
            "cudagraphable", custom.get("cudagraphable")
        )
        custom["partition_id"] = parent_custom.get(
            "partition_id", custom.get("partition_id")
        )
        node.meta["custom"] = custom
        moved += 1
    if moved:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"cudagraph: co-located {moved} getitem(s) with their parent.")


def cudagraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    require_full: bool = False,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
    min_capture_size: int = 1,
) -> torch.fx.GraphModule:
    """Capture the joint graph with CUDA graphs by partitioning it into
    capture-safe subgraphs.

    The single CUDA graph engine; a full CUDA graph is the special case where the
    whole graph is one subgraph. It carves out eager regions (cudagraph-incompatible
    ops -- MoE ``_grouped_mm`` on sm_90, EP token-routing, unpinned CPU<->CUDA
    copies) from capture-safe regions, then:

    - **0 eager regions** -> capture the whole graph (:func:`full_cudagraph_pass`).
    - **>=1 eager region**: if ``require_full`` (``cudagraph_mode='full'``),
      **raise**. Otherwise capture each safe subgraph in its own CUDA graph and run
      the eager regions between replays (enables CUDA graph for DeepSeek-V3 on H100).

    If nothing is capturable, the graph is returned un-wrapped (valid no-op). Capture
    errors at runtime are not swallowed: they usually mean a gap in the unsafe-op
    predicate -- fix it (or use ``cudagraph_mode='off'``) rather than run eager.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example inputs (used by the single-subgraph full capture).
        require_full: When True (explicit ``cudagraph_mode='full'``), raise if the
            graph is not one capturable subgraph instead of partitioning.
        static_input_indices: Top-level input indices with stable addresses
            (params/buffers -- the leading ``num_static_inputs`` flat inputs).
        tensor_input_indices: Top-level tensor input indices (single-subgraph full
            capture only; per-subgraph indices are derived from node metadata).
        min_capture_size: Capturable subgraphs with fewer than this many nodes run
            eager instead (not worth a separate CUDA graph + per-step copy-in).
            Default 1 captures everything.
    """
    # Carve out the tainted module subtrees and assign contiguous-run subgraph ids
    # in a single pass (the predicate is the hot path, so it runs once per node).
    node_to_subgraph_id, subgraph_is_eager = _compute_partition(
        gm, min_capture_size=min_capture_size
    )
    # A getitem indexes a multi-output node whose value is a tuple/list (e.g. a
    # regional-inductor-compiled flex_attention call), which must never cross a
    # capture boundary -- CUDAGraphWrapper rejects list/tuple inputs. Pull every
    # getitem into its parent's subgraph so the tuple/list stays inside one submodule.
    _colocate_getitems(gm, node_to_subgraph_id)
    tlparse_log_graph_pass(
        gm,
        graph_name="cudagraph_partition",
    )

    if not any(subgraph_is_eager.values()):
        # The whole graph is one capturable subgraph -> full CUDA graph.
        logger.info("cudagraph: whole graph is one subgraph; applying full cudagraph.")
        return full_cudagraph_pass(
            gm,
            example_inputs,
            is_forward=True,
            static_input_indices=static_input_indices,
            tensor_input_indices=tensor_input_indices,
        )

    if require_full:
        # Fail loudly rather than partition: the 'full' pipeline kept passes
        # (bucketing, custom_codegen) that assume a single capture.
        raise ValueError(
            "compile.cudagraph_mode='full' but the graph is not one capturable "
            "subgraph: it contains cudagraph-incompatible ops/blocks (e.g. MoE "
            "_grouped_mm on sm_90, EP token-routing splits, data-dependent shapes; "
            "see the warnings above). Use cudagraph_mode='auto' (best-effort: captures "
            "the safe regions and runs the rest eagerly)."
        )

    if all(subgraph_is_eager.values()):
        # Every subgraph is eager: skip the split and return the original graph
        # (auto -> no cudagraph fallback).
        logger.info("cudagraph: nothing capturable; running without cudagraph.")
        return gm

    # Split into per-id submodules, preserving original node order so in-place
    # mutations (e.g. MoE counter copy_) and collectives keep their relative
    # ordering across the eager/cudagraph boundary.
    split_gm = split_module(
        gm,
        gm,
        lambda n: node_to_subgraph_id[n],
        keep_original_order=True,
    )

    # Classify submodules; wrap each capture-safe one in its own cudagraph.
    placeholder_index = {
        n: i
        for i, n in enumerate(n for n in split_gm.graph.nodes if n.op == "placeholder")
    }
    # _arg_is_static relies on split_module preserving top-level input order/count
    # (placeholder position i == original input i); assert it rather than silently
    # misclassify static vs copy-in inputs.
    num_orig_placeholders = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    assert len(placeholder_index) == num_orig_placeholders, (
        f"split_module changed the top-level input count "
        f"({len(placeholder_index)} != {num_orig_placeholders}); "
        f"static-input classification would be unreliable."
    )

    # split_module names each submodule "submod_<id>" with the subgraph id we
    # returned, so we recover its kind by parsing the name (assumes no
    # partition_affix, which we don't pass).
    def _submod_is_cudagraphable(call_node: torch.fx.Node) -> bool:
        subgraph_id = int(call_node.target.removeprefix("submod_"))
        return not subgraph_is_eager[subgraph_id]

    call_module_nodes = [n for n in split_gm.graph.nodes if n.op == "call_module"]
    call_cudagraph_module_nodes = [
        n for n in call_module_nodes if _submod_is_cudagraphable(n)
    ]

    static_set = set(static_input_indices or [])

    def _arg_is_static(arg: Any) -> bool:
        """A subgraph input is static (stable address, no copy-in) if it is a
        param/buffer placeholder or the output of another cudagraph subgraph."""
        if not isinstance(arg, torch.fx.Node):
            return False
        if arg.op == "placeholder":
            return placeholder_index.get(arg, -1) in static_set
        producer = _producer_call_node(arg)
        return producer is not None and producer in call_cudagraph_module_nodes

    for call_node in call_cudagraph_module_nodes:
        submod = getattr(split_gm, call_node.target)
        submod_placeholders = [n for n in submod.graph.nodes if n.op == "placeholder"]
        call_args = call_node.args

        seg_static_indices: list[int] = []
        seg_tensor_indices: list[int] = []
        for j, (arg, ph) in enumerate(zip(call_args, submod_placeholders)):
            val = ph.meta.get("val")
            if isinstance(val, torch.Tensor):
                seg_tensor_indices.append(j)
                if _arg_is_static(arg):
                    seg_static_indices.append(j)

        # example_inputs is unused here (the wrapper only consults it when
        # tensor_input_indices is None, which we always provide), so pass ().
        submod.forward = CUDAGraphWrapper(
            submod.forward,
            (),
            tuple(seg_static_indices),
            tensor_input_indices=seg_tensor_indices,
        )

    num_cudagraph_subgraphs = len(call_cudagraph_module_nodes)
    num_eager_subgraphs = len(call_module_nodes) - num_cudagraph_subgraphs
    logger.info(
        f"Applied piecewise cudagraph: {num_cudagraph_subgraphs} cudagraph subgraph(s), "
        f"{num_eager_subgraphs} eager subgraph(s)."
    )
    return split_gm
