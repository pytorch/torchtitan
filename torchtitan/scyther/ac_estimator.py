import math
import os
import sys
import warnings
from collections import OrderedDict
from dataclasses import astuple, dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
from runtime_estimator import RuntimeEstimator
from torch import nan, nn, UntypedStorage
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.composite_compliance import (
    is_inplace,
    is_inplace_view_fn,
    is_view_fn,
)
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten

try:
    from torch.utils.checkpoint import SAC_IGNORED_OPS  # type: ignore
except ImportError:
    from torch.utils.checkpoint import _ignored_ops as SAC_IGNORED_OPS  # type: ignore

aten = torch.ops.aten

_ADDITIONAL_IGNORED_OPS = {
    aten.lift_fresh.default,  # type: ignore[attr-defined]
    torch.ops.profiler._record_function_exit._RecordFunction,  # type: ignore[attr-defined]
    aten.clone.default,  # type: ignore[attr-defined] # seems needed for torch.compile
}
OPS_TO_ALWAYS_SKIP = SAC_IGNORED_OPS | _ADDITIONAL_IGNORED_OPS
# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


def _get_untyped_storages(t: torch.Tensor) -> Set[torch.UntypedStorage]:
    unflattened_tensors = [t]
    flattened_tensor_storages = set()
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
    return flattened_tensor_storages


# Based on: https://github.com/fairinternal/xformers/blob/0ded5697a2ea15711ce45131002d04e72053cc6d/xformers/checkpoint.py#L62
@dataclass
class _ACMetadata:
    func: Any
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: Tuple[int, ...]
    inplace_info: Tuple[int, ...]
    is_view_like: bool
    is_rand_op: bool


@dataclass
class _ACModMetadata:
    start_idx: int
    force_store_random: bool
    ac_metadata: List[_ACMetadata]


@dataclass
class ACStats:
    func_names: List[str]
    runtimes: List[float]
    memory: List[int]
    view_like_ops: List[int]
    rand_ops: List[int]
    inplace_ops: List[Tuple[int, int]]
    force_store_random: bool


class MSPS(NamedTuple):
    func_names: Set[str]
    op_idx: int
    memory: int
    runtime: float
    msps: float


def _display_stats_tabular(headers: List[str], table_data: List[List[Any]]):
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError("Please install tabulate.") from err

    # Use tabulate to print the table
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


@dataclass
class ACTradeOffStats:
    n_segments: int
    slopes: List[float]
    intercepts: List[float]
    fit_breaks: List[float]
    tradeoff_curve: OrderedDict[float, float]
    ac_memory: int
    ac_runtime: float


@dataclass
class GreedyOrderMeta:
    recomputed_ops: Set[int]
    stored_ops: Set[int]
    inplace_op_groups: Dict[int, Set[int]]
    msps_meta: List[MSPS]


class SACEstimator(TorchDispatchMode):
    def __init__(self):
        self.ac_mod_stats: Dict[str, ACStats] = {}
        self.ac_mod_tradeoff_stats: Dict[str, ACTradeOffStats] = {}
        self.ac_mod_greedy_order_meta: Dict[str, GreedyOrderMeta] = {}
        self._mod_tracker = ModTracker()
        self._ac_metadata: List[_ACMetadata] = []
        self._ac_mod_metadata: Dict[str, _ACModMetadata] = {}
        self._leaf_modules: Set[str] = set()

    def _pre_fw_hook(self, mod: nn.Module, inputs: Any) -> None:
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        assert mod_fqn is not None
        num_children = sum(1 for _ in mod.children())
        if num_children > 0:
            force_store_random = self._get_force_store_random(inputs)
            self._ac_mod_metadata[mod_fqn] = _ACModMetadata(
                start_idx=len(self._ac_metadata),
                force_store_random=force_store_random,
                ac_metadata=[],
            )
        else:
            self._leaf_modules.add(mod_fqn)

    def _post_fw_hook(self, mod: nn.Module, inputs: Any, outputs: Any) -> None:
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        assert mod_fqn is not None
        if mod_fqn in self._leaf_modules:
            return
        else:
            self.ac_mod_stats[mod_fqn] = self._get_ac_stats(
                data=self._ac_mod_metadata[mod_fqn].ac_metadata,
                force_store_random=self._ac_mod_metadata[mod_fqn].force_store_random,
            )
            self.ac_mod_greedy_order_meta[mod_fqn] = self._get_greedy_order_meta(
                self.ac_mod_stats[mod_fqn]
            )

    def _get_force_store_random(self, inputs: Any) -> bool:
        flat_inputs, _ = tree_flatten(inputs)
        return all(not isinstance(x, torch.Tensor) for x in flat_inputs)

    def _get_ac_stats(
        self, data: List[_ACMetadata], force_store_random: bool
    ) -> ACStats:
        # remove aten.detach.default from the list of ops because autograd
        # inserts those during backward and it breaks the fwd-bwd alignment
        data = [x for x in data if x.func not in OPS_TO_ALWAYS_SKIP]

        (
            ops,
            runtimes_,
            memory_,
            new_ids,
            _,
            inplace_ops_,
            view_like_ops_,
            rand_ops_,
        ) = zip(*[astuple(x) for x in data], strict=True)
        runtimes = list(runtimes_)
        memory = list(memory_)
        func_names = [op._overloadpacket.__name__ for op in ops]
        view_like_ops = [i for i, x in enumerate(view_like_ops_) if x]
        rand_ops = [i for i, x in enumerate(rand_ops_) if x]

        # remap the inplace indices as we have removed OPS_TO_ALWAYS_SKIP
        # FIXME @sanketpurandare: Fix this by changing the parent of the inplace-op
        # to itself if the original parent is in OPS_TO_ALWAYS_SKIP.
        try:
            inplace_ops = [tuple(map(new_ids.index, x)) for x in inplace_ops_ if x]
        except ValueError as err:
            raise ValueError(
                f"The remapping of inplace ops failed since one of the inplace op parents"
                f" must have been present in {OPS_TO_ALWAYS_SKIP}"
            ) from err

        # the last operation is always stored as the output of the checkpoint
        # block, so we can avoid recomputing it. We set the memory to zero
        # instead of adding a new constraint because we want both the 0 and 1
        # endpoints for memory_budget to be valid
        # FIXME @sanketpurandare: this heuristic for finding the last non-view non-inplace op
        # might not always be correct, which would yield suboptimal policies
        last_op = len(ops) - 1
        skip_ops_ = set(view_like_ops) | set({x[0] for x in inplace_ops})
        reversed_skip_ops = sorted(skip_ops_, reverse=True)
        for op in reversed_skip_ops:
            if op == last_op:
                last_op -= 1

        memory[last_op] = 0

        return ACStats(
            func_names=func_names,
            runtimes=runtimes,
            memory=memory,
            view_like_ops=view_like_ops,
            rand_ops=rand_ops,
            inplace_ops=inplace_ops,
            force_store_random=force_store_random,
        )

    def _get_inplace_metadata(
        self, func, out_storages: Set[UntypedStorage]
    ) -> Tuple[int, Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        # 1. Get the current index of the metadata obtained so far
        curr_idx = len(self._ac_metadata)
        # 2. Get the set of active modules that are not leaf
        active_mod_fqns: Set[str] = {
            par for par in self._mod_tracker.parents if par not in self._leaf_modules
        }
        # 3. Output ids are the identifies of the storage objects corresponding to the tensors
        output_ids = tuple((id(st) for st in out_storages))
        # 4. If the function is not inplace, return
        if not is_inplace(func):
            return curr_idx, output_ids, {mod_fqn: () for mod_fqn in active_mod_fqns}

        op_id = curr_idx
        # 5. Initialize the parent op ids of the inplace op for each of the active modules
        mod_op_parent_ids: Dict[str, int] = {mod_fqn: -1 for mod_fqn in active_mod_fqns}
        for i, d in enumerate(self._ac_metadata):
            # 6. Find the first occurence of a tensor corresponding to each module that
            # shares the same storage as the current tensor
            past_output_ids = d.output_ids
            if output_ids in past_output_ids:
                for mod_fqn, op_parent_id in mod_op_parent_ids.items():
                    if op_parent_id == -1:
                        if acm_stats := self._ac_mod_metadata.get(mod_fqn, None):
                            if i >= acm_stats.start_idx:
                                mod_op_parent_ids[mod_fqn] = i
                        else:
                            assert mod_fqn == "Global"
                            mod_op_parent_ids[mod_fqn] = i
        # 7. If no parent tensor is found, then it's probably an inplace op on the arguments
        # so one can just store the current-op id as parent id
        for mod_fqn, op_parent_id in mod_op_parent_ids.items():
            if op_parent_id < 0:
                mod_op_parent_ids[mod_fqn] = op_id
        mod_inplace_info = {
            mod_fqn: (op_id, mod_op_parent_ids[mod_fqn]) for mod_fqn in active_mod_fqns
        }
        return curr_idx, output_ids, mod_inplace_info

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs or {})
        # 1. Get the runtime estimate
        op_time = RuntimeEstimator._inductor_estimate(func, args, kwargs, out)
        flat_outs, _ = tree_flatten(out)
        out_storages: Set[UntypedStorage] = set()
        devices: Set[torch.device] = set()
        for o in flat_outs:
            if isinstance(o, torch.Tensor):
                out_storages.update(_get_untyped_storages(o))
                devices.add(o.device)
        # 2. Get the memory consumed by output
        nbytes = sum(st.nbytes() for st in out_storages)
        assert (
            len(devices) <= 1
        ), f"{func.__name__}'s output has more than 1 device types {devices}"
        if devices and next(iter(devices)).type == "cuda":
            nbytes = math.ceil(nbytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        # 3. Get the current operator index, output storage identifiers and inplace metadata
        curr_idx, output_ids, mod_inplace_info = self._get_inplace_metadata(
            func, out_storages
        )
        # 4. Determine if the function is in-place, random-op or a view-like
        is_view_like = is_view_fn(func) or is_inplace_view_fn(func)
        is_rand_op = torch.Tag.nondeterministic_seeded in func.tags
        is_inplace_fn = is_inplace(func)
        if is_view_like or is_inplace_fn:
            nbytes = 0
        # sdpa has non-deterministic seed, but might be deterministic
        # if no dropout is applied
        if func.overloadpacket.__name__ == "_scaled_dot_product_flash_attention":
            is_rand_op = kwargs.get("dropout_p", 0) != 0
        # 5. Create metadata information per active non-leaf module
        for mod_fqn in self._mod_tracker.parents:
            if mod_fqn in self._leaf_modules:
                continue
            acm = _ACMetadata(
                func=func,
                time_taken=op_time,
                memory_used=nbytes,
                curr_idx=curr_idx,
                output_ids=output_ids,
                inplace_info=mod_inplace_info[mod_fqn],
                is_view_like=is_view_like,
                is_rand_op=is_rand_op,
            )
            if acm_stats := self._ac_mod_metadata.get(mod_fqn, None):
                acm_stats.ac_metadata.append(acm)
            else:
                assert (
                    mod_fqn == "Global"
                ), f"Module {mod_fqn} not found in AC Mod Stats"
                self._ac_metadata.append(acm)

        return out

    def _get_greedy_order_meta(self, ac_stats: ACStats) -> GreedyOrderMeta:
        # An inplace-op group is a set of inplace-ops that operate on the same underlying tensor storage.
        # 1. inplace_op_groups: A dictionary from the top-most parent of inplace-ops to the inplace-ops in the group
        #   The top-most op can itself be an inplace-op or can be a non-inplace op.
        # 2. inplace_op_to_group_head: A dictionary that maps all the inplace-ops to their respective group heads.
        inplace_op_groups: Dict[int, Set[int]] = {}
        inplace_op_to_group_head: Dict[int, int] = dict(ac_stats.inplace_ops)

        # 1. Random ops are stored by if force_store_random is set
        # 2. View-like ops are recomputed by default
        # 3. For in-place ops we create a group, recording all the parent-child relations
        #   a) If the head of this group is an inplace op, then we have to store the entire group.
        #   b) If any op in the group is random and force_store_random is set, then entire group will be stored.
        #   c) If none of ops in the group are random and the head of the group is not an in-place op, then
        #       this group can be considered for recomputation in its entireity
        stored_ops: Set[int] = set()
        recomputed_ops: Set[int] = set()
        for op_idx in range(len(ac_stats.memory)):
            if op_idx in ac_stats.view_like_ops:
                # Case 2.
                recomputed_ops.add(op_idx)
                continue
            store = False
            if op_idx in ac_stats.rand_ops and ac_stats.force_store_random:
                # Case 1.
                store = True

            if group_head_idx := inplace_op_to_group_head.get(op_idx, None):
                if op_idx == group_head_idx:
                    # Case 3.(a)
                    stored_ops.add(op_idx)
                    inplace_op_groups[op_idx] = {op_idx}
                else:
                    inplace_op_groups[group_head_idx].add(op_idx)
                    if store and group_head_idx not in stored_ops:
                        # Case 3.(b)
                        stored_ops.add(group_head_idx)
            elif store:
                stored_ops.add(op_idx)

        # The potential recompute candidates are populated as:
        # 1) The in-place op group heads that are not stored
        # 2) The non-inplace ops that are neither stored nor recomputed by default
        recompute_candidates: Set[int] = set()
        for op_idx in inplace_op_groups.keys():
            if op_idx not in stored_ops:
                recompute_candidates.add(op_idx)
        for op_idx in range(len(ac_stats.memory)):
            if (
                (op_idx not in recomputed_ops)
                and (op_idx not in inplace_op_to_group_head)
                and (op_idx not in stored_ops)
            ):
                recompute_candidates.add(op_idx)

        # We define msps for a recomp candidate as the ratio of memory/runtime aka memory savings per second
        msps_meta: List[MSPS] = []
        for cand in recompute_candidates:
            if cand in inplace_op_groups:
                mem = sum(
                    (ac_stats.memory[op_idx] for op_idx in inplace_op_groups[cand])
                )
                runtime = sum(
                    (ac_stats.runtimes[op_idx] for op_idx in inplace_op_groups[cand])
                )
                func_names = {
                    ac_stats.func_names[op_idx] for op_idx in inplace_op_groups[cand]
                }
            else:
                mem = ac_stats.memory[cand]
                runtime = ac_stats.runtimes[cand]
                func_names = {ac_stats.func_names[cand]}
            msps = (mem / runtime) if runtime > 0 else sys.float_info.max
            msps_meta.append(MSPS(func_names, cand, mem, runtime, msps))
        # We choose canidates to be recomputed based on increasing msps
        msps_meta.sort(key=lambda x: x.msps, reverse=True)
        return GreedyOrderMeta(recomputed_ops, stored_ops, inplace_op_groups, msps_meta)

    def _get_ac_tradeoff_stats(
        self,
        ac_stats: ACStats,
        greedy_order_meta: GreedyOrderMeta,
        n_segments: int = 2,
        save_tradeoff_graph: bool = False,
        filename: str = "ac_tradeoff",
    ) -> ACTradeOffStats:
        try:
            import numpy
            import pwlf
        except ImportError as err:
            raise ImportError("Please install pwlf package.") from err

        stored_ops, recomputed_ops, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.msps_meta,
        )
        # Intitialize the discarded memory and recomputation runtime to sum of already chosen recomputed_ops
        discarded_mem = sum((ac_stats.memory[op_idx] for op_idx in recomputed_ops))
        recomp_runtime = sum((ac_stats.runtimes[op_idx] for op_idx in recomputed_ops))
        # Initialize the max recomputation time and total recomputation memory
        ac_runtime = sum(ac_stats.runtimes)
        ac_memory = sum(ac_stats.memory)
        # tradeoff curve stores the ratio of the dicarded memory to total memory vs the runtime penalty to total runtime incurred
        delta = 1e-2
        tradeoff_curve = OrderedDict()
        tradeoff_curve[(discarded_mem / ac_memory) + delta] = (
            recomp_runtime / ac_runtime
        )
        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            tradeoff_curve[(discarded_mem / ac_memory) + delta] = (
                recomp_runtime / ac_runtime
            )
        # Finally, we add the memory and recomputation time of the always stored ops
        discarded_mem += sum((ac_stats.memory[op_idx] for op_idx in stored_ops))
        recomp_runtime += sum((ac_stats.runtimes[op_idx] for op_idx in stored_ops))
        tradeoff_curve[(discarded_mem / ac_memory) + delta] = (
            recomp_runtime / ac_runtime
        )
        x_ = list(tradeoff_curve.keys())
        y_ = list(tradeoff_curve.values())
        # We shift the y values to left and x values to right
        # TODO: Write a better explanation why this needs to be done
        x = x_[: len(x_) - 1]
        y = y_[1:]
        tradeoff_pwlf = pwlf.PiecewiseLinFit(x, y)
        # Fit a piecewise linear function to the tardeoff curve
        n_segments = max(min(len(x) - 2, n_segments), 1)
        tradeoff_pwlf.fit(n_segments=n_segments)

        # save prediction graph
        def save_prediction_graph(
            pwlf_: pwlf.PiecewiseLinFit, x: List[float], y: List[float], filename: str
        ) -> None:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
            except ImportError as err:
                raise ImportError() from err
            # predict for the determined points
            xHat = np.linspace(min(x), max(x), num=10000)
            yHat = pwlf_.predict(xHat)

            # plot the results
            plt.figure()
            plt.plot(x, y, "o", label="Shifted")
            plt.plot(xHat, yHat, "-", label="Predicted")
            plt.plot(x_, y_, "x", label="Original")
            plt.ylabel("Recomp time / Total recomp time")
            plt.xlabel("Memory discarded / Total memory")
            plt.legend()
            plt.title(f"{filename}")
            plt.suptitle(
                f"Total Memory = {ac_memory} B Total Runtime = {ac_runtime:.4f} ms",
                fontsize=10,
            )
            folder_name = "tradeoff_graphs"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            # Save the plots in the folder
            plt.savefig(os.path.join(folder_name, f"{filename}.png"))

        if save_tradeoff_graph:
            save_prediction_graph(tradeoff_pwlf, x, y, filename)
        # Obtain the slopes, intercepts and breakpoints of the fitted piecewise linear functions
        slopes = tradeoff_pwlf.calc_slopes().tolist()
        assert isinstance(tradeoff_pwlf.intercepts, numpy.ndarray) and isinstance(
            tradeoff_pwlf.fit_breaks, numpy.ndarray
        )
        intercepts = tradeoff_pwlf.intercepts.tolist()
        fit_breaks = tradeoff_pwlf.fit_breaks.tolist()
        return ACTradeOffStats(
            n_segments=n_segments,
            slopes=slopes,
            intercepts=intercepts,
            fit_breaks=fit_breaks,
            tradeoff_curve=tradeoff_curve,
            ac_memory=ac_memory,
            ac_runtime=ac_runtime,
        )

    def display_ac_stats(self, ac_stats: ACStats, print_tabular: bool) -> None:
        print(
            f"Total Memory: {sum(ac_stats.memory)} B Total Runtime: {sum(ac_stats.runtimes)} ms"
            f" Store Random: {ac_stats.force_store_random}"
        )
        table_data = []
        op_parent = dict(ac_stats.inplace_ops)
        for i, fn_name in enumerate(ac_stats.func_names):
            row = [
                str(i),
                fn_name,
                f"{ac_stats.runtimes[i]:.4f}",
                str(ac_stats.memory[i]),
                str(i in ac_stats.view_like_ops),
                str(i in ac_stats.rand_ops),
                str(op_parent.get(i, None)),
            ]
            table_data.append(row)
        # Define headers
        headers = [
            "Op Idx",
            "Op Name",
            "Runtimes(ms)",
            "Memory (B)",
            "View-like",
            "Random",
            "In-place",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def display_ac_tradeoff_stats(
        self,
        greedy_order_meta: GreedyOrderMeta,
        ac_stats: ACStats,
        print_tabular: bool = False,
    ):
        table_data = []
        total_memory, total_runtime = sum(ac_stats.memory), sum(ac_stats.runtimes)
        discarded_mem = recomp_runtime = 0

        def append_row(
            op_indices: Set[int],
            func_names: Set[str],
            msps: Optional[float] = None,
            stored: Optional[bool] = False,
            recomputed: Optional[bool] = False,
        ):
            row = [
                str(op_indices),
                str(func_names),
                f"{discarded_mem / total_memory:.4f}",
                str(discarded_mem),
                f"{recomp_runtime / total_runtime:.4f}",
                str(recomp_runtime),
                f"{msps:.2e}" if msps is not None else str(nan),
                str(stored),
                str(recomputed),
            ]
            table_data.append(row)

        stored_ops, recomputed_ops, inplace_op_groups, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.inplace_op_groups,
            greedy_order_meta.msps_meta,
        )

        for op_idx in recomputed_ops:
            op_indices: Set[int] = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            discarded_mem += sum(ac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(ac_stats.runtimes[i] for i in op_indices)
            func_names = {ac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, recomputed=True)

        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            op_indices: Set[int] = {cand.op_idx}
            if cand.op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[cand.op_idx])
            append_row(op_indices, cand.func_names, msps=cand.msps)

        for op_idx in stored_ops:
            op_indices: Set[int] = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            discarded_mem += sum(ac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(ac_stats.runtimes[i] for i in op_indices)
            func_names = {ac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, stored=True)

        headers = [
            "Op Id(s)",
            "Op Name(s)",
            "Discarded Mem (%)",
            "Discarded Mem (B)",
            "Recomp time (%)",
            "Recomp time (ms)",
            "MSPS",
            "Always Stored",
            "Always Recomputed",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def pwlf_ac_tradeoff_stats(
        self,
        n_segments: int = 2,
        save_tradeoff_graphs: bool = False,
    ):
        for mod_fqn in self.ac_mod_stats.keys():
            self.ac_mod_tradeoff_stats[mod_fqn] = self._get_ac_tradeoff_stats(
                ac_stats=self.ac_mod_stats[mod_fqn],
                greedy_order_meta=self.ac_mod_greedy_order_meta[mod_fqn],
                n_segments=n_segments,
                save_tradeoff_graph=save_tradeoff_graphs,
                filename=mod_fqn,
            )

    def display_modulewise_ac_stats(
        self, depth: int = 2, print_tabular: bool = False
    ) -> None:
        for mod_fqn, ac_stats in self.ac_mod_stats.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module: {mod_fqn}")
            self.display_ac_stats(ac_stats, print_tabular)
            print(f"AC Trade-off for Module: {mod_fqn} MSPS = Memory/Runtime")
            self.display_ac_tradeoff_stats(
                self.ac_mod_greedy_order_meta[mod_fqn], ac_stats, print_tabular
            )

    def __enter__(self):
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "SAC Estimator should be called in FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=self._pre_fw_hook,
            post_fw_hook=self._post_fw_hook,
        )
        self._mod_tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args: Any):
        self._mod_tracker.__exit__(*args)
        super().__exit__(*args)
