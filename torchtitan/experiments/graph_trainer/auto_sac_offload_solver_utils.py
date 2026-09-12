# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator

import torch

# Registers the bucketing::_pre_bucket_* custom ops. Without it, the solver's
# torch.ops.bucketing lookups raise AttributeError.
import torch._inductor.fx_passes.bucketing  # noqa: F401
from torch.utils._runtime_estimation import _IGNORE_OPS

ROUNDING = 512  # the CUDA allocator rounds allocations up to 512B

# States kept per parameter, by optimizer name. A 0 claims the optimizer is
# stateless; amsgrad and SGD momentum add one more, handled by the caller.
STATES_PER_PARAM = {
    "Adam": 2,  # exp_avg, exp_avg_sq
    "AdamW": 2,  # exp_avg, exp_avg_sq
    "SGD": 0,  # stateless unless momentum is set
    "Adagrad": 1,  # sum
    "RMSprop": 1,  # square_avg
    "Adafactor": 1,  # factored second moment
    "Lion": 1,  # exp_avg
}

# Reported when the optimizer is not in STATES_PER_PARAM.
UNKNOWN_OPTIMIZER_BYTES = -1

# ---- storage categories ----
PARAM = "parameter"
GRAD = "gradient"
ACT = "activation"  # forward intermediate that survives into backward
TEMP = "temporary"  # intermediate freed within its own (fwd or bwd) region
INPUT = "input"
BWD_TEMP = "backward_temporary"

# Estimate modes, named to match the upstream RuntimeEstimator.
COST_MODEL = "operator-level-cost-model"
BENCHMARK = "operator-level-benchmark"
INTERPRETER = "operator-level-interpreter"

# Below this the benchmark is launch-bound and under-reports the link.
_MIN_SATURATING_BYTES = 64 * 1024 * 1024  # 64 MB
# Share of free device memory the bandwidth benchmark may take.
_FREE_MEM_FRACTION = 0.9

# Ops whose output aliases their first tensor input, by overloadpacket name.
# Used to repair graphs whose node.meta["val"] lost the aliasing relationship.
_VIEW_OP_NAMES = frozenset(
    {
        "t",
        "transpose",
        "view",
        "_unsafe_view",
        "reshape",
        "unsqueeze",
        "squeeze",
        "slice",
        "select",
        "expand",
        "permute",
        "as_strided",
        "alias",
        "detach",
        "narrow",
        "unfold",
        "split",
        "split_with_sizes",
        "chunk",
        "movedim",
        "swapaxes",
        "ravel",
        "flatten",
    }
)


def _concrete_bytes(nbytes) -> int:
    """Resolve a storage size to a plain int, including the symbolic sizes MoE
    routing produces. Raises rather than guess, since a size guessed low makes
    the memory budget too generous and the run OOMs.
    """
    if isinstance(nbytes, int):
        return nbytes
    node = getattr(nbytes, "node", None)
    env = getattr(node, "shape_env", None)
    expr = getattr(node, "expr", None)
    if env is not None and expr is not None:
        # the hint is the value observed while tracing
        hint = env.size_hint(expr, allow_none=True)
        if hint is not None:
            return int(hint)
        upper = env.bound_sympy(expr).upper
        if upper.is_finite:
            return int(upper)
    raise RuntimeError(
        f"cannot resolve symbolic storage size {nbytes!r} to a number. This is "
        "an MoE model whose per-expert token counts are data dependent, so the "
        "shapes are unbacked and the peak cannot be estimated without an "
        "assumption about expert capacity. Bound the routing (a fixed capacity "
        "factor makes the shapes static) or use a non-ILP memory policy."
    )


def get_size(t: torch.Tensor) -> int:
    """Allocation size of the tensor's underlying storage, rounded like the
    allocator. A view has a small numel but can reference a much larger storage.
    """
    nbytes = _concrete_bytes(t.untyped_storage().nbytes())
    return math.ceil(nbytes / ROUNDING) * ROUNDING


def _is_costable_op(node: torch.fx.Node) -> bool:
    """Whether to apply the roofline to this node, mirroring the upstream
    RuntimeEstimator: everything except _IGNORE_OPS and getitem.

    Unknown ops are not dropped to zero -- they still contribute transfer time,
    and get_compute_time falls back to 0 when the flop registry lacks them.
    """
    if node.op != "call_function":
        return False
    target = node.target
    # getitem only indexes a multi-output result; no kernel cost
    if target is operator.getitem:
        return False
    if isinstance(target, torch._ops.OpOverload):
        return target._overloadpacket not in _IGNORE_OPS
    return True
