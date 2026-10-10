# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complete local shared-FFN measurements with prepared MXFP8 weights.

T = tokens, D = model width, F = hidden width. Weight preparation and
collectives are outside this scope. All activation quantization, training
saves, dx, and both weight gradients are included.
"""

import argparse
import gc
import hashlib
import json
import math
import statistics
import warnings
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.linear import (
    InputActivationFormatForBackward,
    MXFP8Linear,
)
from torchtitan.quantization.mxfp8.tensor import _quantize_mxfp8_weight
from torchtitan_recipes.overrides._dsv3_shared_expert.autograd import (
    quantize,
    scaled_mm,
)
from torchtitan_recipes.overrides._dsv3_shared_expert.ops import (
    shared_expert_backward_op,
    shared_expert_forward_op,
)
from torchtitan_recipes.overrides.fused_dsv3_shared_expert import (
    fused_dsv3_shared_expert,
)
from torchtitan_recipes.overrides.fused_swiglu import (
    FusedSwiGLU,
    silu_and_mul_backward_op,
    silu_and_mul_op,
)


MODES = {
    "native": (False, False),
    "forward_fused": (True, False),
    "backward_fused": (False, True),
    "both_fused": (True, True),
}


def shared_expert_config(
    *,
    input_save_format: InputActivationFormatForBackward = "bf16",
    hidden_save_format: InputActivationFormatForBackward = "mxfp8",
):
    return FeedForward.Config(
        w13=MXFP8Linear.Config(
            in_features=7168,
            out_features=2048,
            num_linears=2,
            input_activation_format_for_backward=input_save_format,
        ),
        w2=MXFP8Linear.Config(
            in_features=2048,
            out_features=7168,
            input_activation_format_for_backward=hidden_save_format,
        ),
        activation_fn=FusedSwiGLU.Config(),
    )


def make_case(
    seed,
    mode,
    *,
    input_save_format: InputActivationFormatForBackward = "bf16",
    hidden_save_format: InputActivationFormatForBackward = "mxfp8",
    requires_grad=(True, True, True),
    grad_dtype=None,
    prepared_weights=True,
):
    """Nonzero production-shape inputs; stand in for FSDP weight preparation."""
    torch.manual_seed(seed)
    config = shared_expert_config(
        input_save_format=input_save_format, hidden_save_format=hidden_save_format
    )
    forward, backward = MODES[mode]
    if forward or backward:
        config = fused_dsv3_shared_expert(
            config, forward_quant=forward, backward_quant=backward
        )
    with torch.device("cuda"):
        module = config.build().bfloat16()
    for linear, needs_grad in zip((module.w13, module.w2), requires_grad[1:]):
        sharded_weight = linear.weight
        with torch.no_grad():
            sharded_weight._tensor.normal_(0, 0.02)
            if prepared_weights:
                linear.weight = nn.Parameter(
                    _UnshardedFSDPTensor(
                        sharded_weight._tensor,
                        sharded_weight._build_operands(sharded_weight._tensor),
                    ),
                    requires_grad=needs_grad,
                )
        linear.weight.requires_grad_(needs_grad)
        if grad_dtype is not None:
            linear.weight.grad_dtype = grad_dtype
    x_TD = torch.randn(
        1,
        4096,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=requires_grad[0],
    )
    dy_TD = torch.randn_like(x_TD)
    return module, x_TD, dy_TD


def assert_bits_equal(expected, actual):
    if expected is None or actual is None:
        assert expected is actual
        return
    assert (expected.shape, expected.dtype) == (actual.shape, actual.dtype)
    expected_bits = (
        expected.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    actual_bits = (
        actual.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    assert torch.equal(expected_bits, actual_bits), (
        f"{int((expected_bits != actual_bits).sum())} mismatched bytes in "
        f"{tuple(expected.shape)} {expected.dtype}"
    )


def _storage_bytes(tensors):
    storages = {
        tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
        for tensor in tensors
    }
    return sum(storages.values())


def _saved_tensors(output):
    tensors, visited, pending = [], set(), [output.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for tensor in getattr(node, "saved_tensors", ()):
            if isinstance(tensor, _UnshardedFSDPTensor):
                operands = tensor.operands
                tensors.extend(
                    (
                        operands.weight_qdata_dgrad_NK,
                        operands.weight_scale_dgrad_swizzled,
                    )
                )
            elif isinstance(tensor, torch.Tensor):
                tensors.append(tensor)
        pending.extend(child for child, _ in node.next_functions)
    return tensors


def _module_functions(mode, phase, banks, *, input_save_format):
    functions, touched, states = [], [], []
    for bank in range(banks):
        module, x, dy = make_case(
            919 + bank * 17, mode, input_save_format=input_save_format
        )
        weights = (module.w13.weight, module.w2.weight)
        state = {}
        states.append(state)

        def invoke(module=module, x=x, dy=dy, weights=weights, state=state):
            if phase == "backward":
                if "output" not in state:
                    state["output"] = module(x)
                output = state["output"]
            else:
                output = module(x)
            if phase == "forward":
                return output
            gradients = torch.autograd.grad(
                output, (x, *weights), dy, retain_graph=phase == "backward"
            )
            return output, gradients

        functions.append(invoke)
        if phase != "backward":
            touched.append(x)
        if phase != "forward":
            touched.append(dy)
        for parameter in weights:
            operands = parameter.operands
            touched.append(operands.weight_qdata_dgrad_NK)
            if phase != "backward":
                touched.append(operands.weight_scale_fprop_swizzled)
            if phase != "forward":
                touched.append(operands.weight_scale_dgrad_swizzled)

    def buffers():
        saved = (
            [tensor for state in states for tensor in _saved_tensors(state["output"])]
            if phase == "backward"
            else []
        )
        return [*touched, *saved]

    return functions, buffers


def _percentile(samples, fraction):
    ordered = sorted(samples)
    index = fraction * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure(factory, *, regime, samples, iterations, profiles=False):
    """Create backward fixtures on the capture stream and time complete cycles."""
    gc.collect()
    torch.cuda.empty_cache()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        functions, buffers = factory()
        for function in functions:
            function()
        touched_bytes = _storage_bytes(buffers())
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if regime == "streaming" and touched_bytes <= 2 * l2_bytes:
            raise ValueError(
                "Streaming requires independently touched banks exceeding twice L2"
            )
        stream.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        operations = iterations * len(functions)
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                for function in functions:
                    function()
        pool = torch.cuda.memory_snapshot(mempool_id=graph.pool(), include_traces=False)
        pool_bytes = sum(segment["total_size"] for segment in pool)
        peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
        for _ in range(10):
            graph.replay()
        elapsed = []
        for _ in range(samples):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed.append(begin.elapsed_time(end) * 1000 / operations)
        report = {
            "median_us": statistics.median(elapsed),
            "p90_us": _percentile(elapsed, 0.9),
            "cv": statistics.pstdev(elapsed) / statistics.mean(elapsed),
            "samples_us": elapsed,
            "banks": len(functions),
            "operations_per_graph": operations,
            "touched_storage_bytes": touched_bytes,
            "graph_pool_reserved_bytes": pool_bytes,
            "working_set_upper_bound_bytes": touched_bytes + pool_bytes,
            "capture_incremental_peak_bytes": peak_bytes,
            "device_l2_bytes": l2_bytes,
        }
        if profiles:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                functions[0]()
                torch.cuda.synchronize()
            events = profile.events()
            assert events is not None
            names = [
                event.name
                for event in events
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            report["launches"] = {"count": len(names), "names": dict(Counter(names))}
    torch.cuda.current_stream().wait_stream(stream)
    del graph, functions, buffers
    return report


def _primitive_functions(direction, fused, banks):
    functions, touched = [], []
    for bank in range(banks):
        torch.manual_seed(919 + bank * 17)
        source = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
        row, _, scale, _ = quantize(source, colwise=False)
        shape = (4096, 7168) if direction == "forward" else (7168, 2048)
        weight = _quantize_mxfp8_weight(
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        qdata = weight.weight_qdata_dgrad_NK
        weight_scale = (
            weight.weight_scale_fprop_swizzled
            if direction == "forward"
            else weight.weight_scale_dgrad_swizzled
        )
        packed = (
            torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
            if direction == "backward"
            else None
        )

        def invoke(
            row=row, scale=scale, qdata=qdata, weight_scale=weight_scale, packed=packed
        ):
            if direction == "forward":
                if fused:
                    return shared_expert_forward_op(row, qdata, scale, weight_scale)
                gate_up = scaled_mm(row, qdata.t(), scale, weight_scale)
                hidden = silu_and_mul_op(gate_up.unflatten(-1, (2, 2048)))
                return hidden, gate_up, *quantize(hidden)
            assert packed is not None
            if fused:
                return shared_expert_backward_op(
                    row, qdata, scale, weight_scale, packed
                )
            hidden_gradient = scaled_mm(row, qdata, scale, weight_scale)
            packed_gradient = silu_and_mul_backward_op(
                hidden_gradient, packed.unflatten(-1, (2, 2048))
            ).flatten(-2)
            return quantize(packed_gradient)

        functions.append(invoke)
        touched.extend((row, qdata, scale, weight_scale))
        if packed is not None:
            touched.append(packed)
    return functions, lambda: touched


@triton.jit
def _copy_control(source, output, N: tl.constexpr):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(
        output + offsets, tl.load(source + offsets, offsets < N, other=0), offsets < N
    )


@triton.jit
def _launch_control(pointer):
    pass


def _memory_functions(elements):
    source = torch.full((elements,), 0.5, device="cuda")
    output = torch.empty_like(source)
    return [
        lambda: _copy_control[(triton.cdiv(elements, 1024),)](source, output, elements)
    ], lambda: [source, output]


def _compute_functions(size):
    source = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    row, _, scale, _ = quantize(source, colwise=False)
    weight = _quantize_mxfp8_weight(source)
    qdata, weight_scale = (
        weight.weight_qdata_fprop_KN,
        weight.weight_scale_fprop_swizzled,
    )
    return [lambda: scaled_mm(row, qdata, scale, weight_scale)], lambda: [
        row,
        scale,
        qdata,
        weight_scale,
    ]


def _controls(samples):
    controls = {}
    for name, regime, elements in (
        ("l2", "warm", 2**23),
        ("hbm", "streaming", 2**26),
    ):
        timing = measure(
            partial(_memory_functions, elements),
            regime=regime,
            samples=samples,
            iterations=20,
        )
        timing["bytes_per_second"] = 2 * elements * 4 / (timing["median_us"] * 1e-6)
        controls[name] = timing
    controls["compute"] = measure(
        partial(_compute_functions, 12288),
        regime="streaming",
        samples=samples,
        iterations=4,
    )
    controls["compute"]["ops_per_second"] = (
        2 * 12288**3 / (controls["compute"]["median_us"] * 1e-6)
    )
    controls["launch"] = measure(
        lambda: ([lambda: _launch_control[(1,)](0)], lambda: []),
        regime="warm",
        samples=samples,
        iterations=100,
    )
    return controls


def _roofline(primitives, controls):
    tokens, dim, hidden = 4096, 7168, 2048
    common_input = tokens * dim
    required_bytes = {
        "forward": (common_input + 2 * hidden * dim) * 33 // 32
        + tokens * hidden * 8
        + 2 * tokens * hidden // 32,
        "backward": (common_input + hidden * dim) * 33 // 32
        + tokens * hidden * 8
        + 4 * tokens * hidden // 32,
    }
    result = {}
    for direction in ("forward", "backward"):
        flop = 2 * tokens * dim * hidden * (2 if direction == "forward" else 1)
        memory_us = (
            required_bytes[direction] / controls["hbm"]["bytes_per_second"] * 1e6
        )
        compute_us = flop / controls["compute"]["ops_per_second"] * 1e6
        bound = max(memory_us, compute_us, controls["launch"]["median_us"])
        timing = primitives["streaming"][direction]["fused"]
        fraction = bound / timing["p90_us"]
        result[direction] = {
            "required_bytes": required_bytes[direction],
            "gemm_ops": flop,
            "hbm_bound_us": memory_us,
            "compute_bound_us": compute_us,
            "bound_us": bound,
            "p90_roof_fraction": fraction,
            "necessary_90_percent_gate": 0.9 <= fraction <= 1.1
            and timing["cv"] <= 0.05,
            "accepted": False,
            "limitation": "A necessary HBM/compute/launch bound; combined SFU/conversion throughput is not certified.",
        }
    return result


def _kernel_metadata():
    from torchtitan_recipes.overrides._dsv3_shared_expert.kernels.backward import (
        launch_swiglu_backward,
    )
    from torchtitan_recipes.overrides._dsv3_shared_expert.kernels.forward import (
        shared_w13_swiglu,
    )

    result = {}
    for direction in ("forward", "backward"):
        functions, buffers = _primitive_functions(direction, True, 1)
        metadata = {}
        if direction == "forward":
            shared_w13_swiglu(
                *buffers(), quantize_output=True, maxnreg=160, kernel_metadata=metadata
            )
        else:
            launch_swiglu_backward(*buffers(), kernel_metadata=metadata)
        result[direction] = metadata
        del functions, buffers
    return result


def main():
    warnings.filterwarnings(
        "error", message="The AccumulateGrad node's stream does not match"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--banks", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument(
        "--input-save-format", choices=("bf16", "mxfp8"), default="bf16"
    )
    args = parser.parse_args()
    if args.samples < 10 or args.iterations < 1 or args.banks < 5 or args.rounds < 1:
        parser.error("Use >=10 samples, >=1 iteration, >=5 banks, and >=1 round")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torchao

    report = {
        "accepted": False,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torchao": torchao.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": str(torch.cuda.get_device_properties(0)),
        "input_shape": [1, 4096, 7168],
        "weight_shapes": [[2, 2048, 7168], [7168, 2048]],
        "dtype": "bfloat16",
        "input_save_format": args.input_save_format,
        "hidden_save_format": "mxfp8",
        "scope": (
            "Complete local shared FFN; prepared weight quantization and collectives excluded; "
            "fresh x/W13/W2 gradients from a random BF16 upstream."
        ),
        "baseline": "Public TorchTitan MXFP8Linear + FusedSwiGLU at the bottom of this stack, not the older fbpackage binary.",
        "module": {},
        "primitives": {},
    }
    source_root = Path(__file__).resolve().parents[1]
    relative_sources = [
        "benchmarks/fused_dsv3_shared_expert.py",
        "torchtitan_recipes/overrides/fused_dsv3_shared_expert.py",
        *[
            f"torchtitan_recipes/overrides/_dsv3_shared_expert/{name}.py"
            for name in (
                "autograd",
                "ops",
                "kernels/forward",
                "kernels/forward_quant",
                "kernels/backward",
                "kernels/quantize",
                "kernels/descriptors",
            )
        ],
    ]
    report["source_sha256"] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in relative_sources
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def parity_values(mode):
        module, x, dy = make_case(919, mode, input_save_format=args.input_save_format)
        output = module(x)
        return (
            output.detach(),
            *torch.autograd.grad(output, (x, module.w13.weight, module.w2.weight), dy),
        )

    reference = parity_values("native")
    for mode in list(MODES)[1:]:
        for expected, actual in zip(reference, parity_values(mode)):
            assert_bits_equal(expected, actual)
    del reference
    report["complete_ffn_bitwise_parity"] = True
    for regime, banks in (("warm", 1), ("streaming", args.banks)):
        report["module"][regime] = {mode: {} for mode in MODES}
        for round_index in range(args.rounds):
            order = list(MODES) if round_index % 2 == 0 else list(reversed(MODES))
            for phase in ("forward", "backward", "combined"):
                for mode in order:
                    timing = measure(
                        partial(
                            _module_functions,
                            mode,
                            phase,
                            banks,
                            input_save_format=args.input_save_format,
                        ),
                        regime=regime,
                        samples=args.samples,
                        iterations=args.iterations,
                        profiles=args.profiles and round_index == 0,
                    )
                    report["module"][regime][mode].setdefault(phase, []).append(timing)
                    print(
                        regime,
                        round_index,
                        mode,
                        phase,
                        round(timing["median_us"], 2),
                        "us",
                        flush=True,
                    )
                    save()
        report["primitives"][regime] = {}
        for direction in ("forward", "backward"):
            report["primitives"][regime][direction] = {}
            for fused in (False, True):
                timing = measure(
                    partial(_primitive_functions, direction, fused, banks),
                    regime=regime,
                    samples=args.samples,
                    iterations=args.iterations,
                    profiles=args.profiles,
                )
                report["primitives"][regime][direction][
                    "fused" if fused else "native"
                ] = timing
                print(
                    regime,
                    direction,
                    "fused" if fused else "native",
                    round(timing["median_us"], 2),
                    "us",
                    flush=True,
                )
                save()
    report["kernel_metadata"] = _kernel_metadata()
    report["controls"] = _controls(args.samples)
    report["roofline"] = _roofline(report["primitives"], report["controls"])
    report["summary_us"] = {
        regime: {
            mode: {
                phase: statistics.median(
                    [sample for run in runs for sample in run["samples_us"]]
                )
                for phase, runs in phases.items()
            }
            for mode, phases in modes.items()
        }
        for regime, modes in report["module"].items()
    }
    save()
    print(
        json.dumps(
            {"summary_us": report["summary_us"], "roofline": report["roofline"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
