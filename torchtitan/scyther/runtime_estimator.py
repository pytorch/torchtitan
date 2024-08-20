import time
from contextlib import nullcontext
from collections import defaultdict
from typing import Callable, Dict, Set

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

aten = torch.ops.aten

# No fall-back kernel needed/exists for view ops
_IGNORE_OPS = {
    aten.lift_fresh,
    aten.t,
    aten.transpose,
    aten.view,
    aten.detach,
    aten._unsafe_view,
    aten.split,
    aten.adjoint,
    aten.as_strided,
    aten.diagonal,
    aten.expand,
    aten.expand_as,
    aten.movedim,
    aten.permute,
    aten.select,
    aten.squeeze,
    aten.mT,
    aten.mH,
    aten.real,
    aten.imag,
    aten.view_as,
    aten.unflatten,
    aten.unfold,
    aten.unbind,
    aten.unsqueeze,
    aten.vsplit,
    aten.hsplit,
    aten.split_with_sizes,
    aten.swapaxes,
    aten.swapdims,
    aten.chunk,
}
# We can ignore benchmarking tensor create ops
_CREATE_OPS = {
    aten.randint,
    aten.randn,
    aten.rand,
    aten.randn_like,
    aten.rand_like,
    aten.randint_like,
    aten.arange,
    aten.ones_like,
    aten.zeros_like,
}

_IGNORE_OPS_EXT = _IGNORE_OPS | _CREATE_OPS

__all__ = ["RuntimeEstimator"]


class RuntimeEstimator(TorchDispatchMode):
    _gpu_memory_bandwidth = get_gpu_dram_gbps()
    _float_types: Set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _no_fallback_kernel = set()
    fake_mode: FakeTensorMode

    def __init__(self):
        self._estimate: Callable
        self._estimate_mode_type: str
        self._mod_tracker = ModTracker()
        self.mod_runtimes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order = []
        self.mod_bw_pre_order = []
        self.mod_fw_post_order = []
        self.mod_bw_post_order = []
        self.total_runtime: float = 0.0

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be umodified
        if torch.Tag.inplace_view in func.tags:
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):
                if cls.fake_mode.is_our_fake(e):
                    if e.dtype in cls._float_types:
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            num_iters = 3
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            cpu_start = time.time()
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                r = None
                r = func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            cpu_end = time.time()
            torch.cuda.synchronize()
            cpu_time = (cpu_end - cpu_start) / 1000
            total_op_time = start_event.elapsed_time(end_event) - cpu_time
            mean_op_time = total_op_time / num_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs, res) -> float:
        assert isinstance(
            cls.fake_mode, FakeTensorMode
        ), "Initialize/Assign FakeTensorMode before using this function"
        mean_op_time = 0.0
        if func._overloadpacket not in _IGNORE_OPS_EXT:
            try:
                res, mean_op_time = cls._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return mean_op_time
            except NotImplementedError:
                cls._no_fallback_kernel.add(func._overloadpacket)
        return mean_op_time

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa
    @classmethod
    def _inductor_estimate(cls, func, args, kwargs, out) -> float:
        def get_num_bytes(t: torch.Tensor) -> int:
            st = t.untyped_storage()
            num_bytes = st.size() * st.element_size()
            return num_bytes

        def get_compute_time(func_packet, args, kwargs, out, out_dtypes):
            if func_packet in flop_registry:
                assert (
                    len(out_dtypes) == 1
                ), f"Only support single out dtype got {out_dtypes}"
                f"{out_dtypes} for {func_packet}"
                dtype = out_dtypes.pop()
                # We can expect to achieve 75% of theoretical peak flops
                factor = 0.75
                # This actually gives peta-FLOPs/s hence multiply by 1e15
                # instead of 1e12 to get the FLOPs/s
                gpu_flops = get_device_tflops(dtype) * 1e15
                flop_count_func = flop_registry[func_packet]
                # We divide by a factor of 2 to get the MACs
                # (multiply and accumulate)
                flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
                # We multiply by 1e9 to get the time in nano seconds
                compute_time = (flop_count / (factor * gpu_flops)) * 1e9
                return compute_time
            return 0.0

        def get_transfer_time(flat_args_kwargs, flat_outs):
            read_bytes = sum(
                get_num_bytes(t)
                for t in flat_args_kwargs
                if isinstance(t, torch.Tensor)
            )
            write_bytes = sum(
                get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
            )
            counted_bytes = read_bytes + write_bytes
            # The GPU memory bandwidth is in GB/s so the transfer time
            # is in nano seconds
            transfer_time = counted_bytes / cls._gpu_memory_bandwidth
            return transfer_time

        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:

            flat_args_kwargs, args_spec = pytree.tree_flatten((args, kwargs))
            flat_outs, out_spec = pytree.tree_flatten(out)
            transfer_time = get_transfer_time(flat_args_kwargs, flat_outs)

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in cls._float_types
            }

            args, kwargs = pytree.tree_unflatten(flat_args_kwargs, args_spec)
            out = pytree.tree_unflatten(flat_outs, out_spec)

            compute_time = get_compute_time(func_packet, args, kwargs, out, out_dtypes)
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6

        return op_time

    def display_modulewise_stats(self, depth: int = 2):
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = func(*args, **kwargs or {})
        # FIXME @sanketpurandare: faltten tensors by desugaring the tensor subclasses
        op_time = self._estimate(func, args, kwargs, res)
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                self.mod_runtimes[par]["bw"] += op_time
            else:
                self.mod_runtimes[par]["fw"] += op_time
        self.total_runtime += op_time
        return res

    def __call__(self, estimate_mode_type: str):
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = RuntimeEstimator._inductor_estimate
        elif estimate_mode_type == "actual":
            return nullcontext()
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self):
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self.total_runtime = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
        )
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args):
        print(
            f"Estimated ({self._estimate_mode_type})"
            f"total_time: {self.total_runtime:.3f} ms"
        )
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
