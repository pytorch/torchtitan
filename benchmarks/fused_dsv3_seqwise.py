# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-graph timings for the fixed DeepSeek V3 fusion specialization.

Reports component forward, retained-graph backward, and forward+backward.
Independent inputs rotate beyond L2. These are not full-model training times.
"""

import argparse
import json
import random
import statistics

import torch
import torch.nn.functional as F

from torchtitan_recipes.overrides.fused_dsv3_seqwise_loss import (
    FusedDSv3SeqwiseLossFunction,
    kernels,
)


def _raw_loss(scores_TE, routing_map_TE):
    frequencies_E = F.normalize(routing_map_TE.float().sum(0), p=1, dim=0) * 256
    return (frequencies_E * F.normalize(scores_TE, p=1, dim=-1).sum(0)).sum()


def _make_functions(reference):
    ids = torch.rand_like(reference).topk(8, dim=-1, sorted=False).indices
    routing_map = torch.zeros_like(reference, dtype=torch.bool).scatter_(-1, ids, True)
    counter = torch.zeros(1, dtype=torch.int32, device=reference.device)

    def native(x):
        return (_raw_loss(x, routing_map),)

    def fused(x):
        return (FusedDSv3SeqwiseLossFunction.apply(x, routing_map, counter),)

    return native, fused, (torch.tensor(0.001 / 4096, device=reference.device),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffers", type=int, default=96)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.buffers < 1 or args.samples < 1:
        parser.error("buffers and samples must be positive")
    torch.manual_seed(7)
    kernels.prepare()
    inputs = [
        torch.randn(4096, 256, device="cuda").sigmoid().requires_grad_()
        for _ in range(args.buffers)
    ]
    native, fused, upstream = _make_functions(inputs[0])
    native_output, fused_output = native(inputs[0]), fused(inputs[0])
    (native_grad,) = torch.autograd.grad(native_output, inputs[0], upstream)
    (fused_grad,) = torch.autograd.grad(fused_output, inputs[0], upstream)
    for expected, actual in zip(
        (*native_output, native_grad), (*fused_output, fused_grad)
    ):
        if not torch.equal(
            expected.reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8)
        ):
            raise AssertionError("bitwise parity must pass before timing")
    del native_output, fused_output

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    saved_backward_outputs = {}
    # Use fresh leaves on the capture stream to avoid stale AccumulateGrad
    # stream references from the parity check above.
    with torch.cuda.stream(stream):
        inputs = [x.detach().requires_grad_() for x in inputs]
        for name, function in (("native", native), ("fused", fused)):
            for phase in ("forward", "backward", "forward_backward"):
                saved_outputs = (
                    [function(x) for x in inputs] if phase == "backward" else None
                )
                if saved_outputs is not None:
                    saved_backward_outputs[name] = saved_outputs

                def run():
                    for index, x in enumerate(inputs):
                        outputs = (
                            saved_outputs[index]
                            if saved_outputs is not None
                            else function(x)
                        )
                        if phase != "forward":
                            torch.autograd.grad(
                                outputs, x, upstream, retain_graph=phase == "backward"
                            )

                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run()
                graphs[name, phase] = graph
        for graph in graphs.values():
            for _ in range(5):
                graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    timings = {key: [] for key in graphs}
    order = list(graphs)
    rng = random.Random(7)
    for _ in range(args.samples):
        rng.shuffle(order)
        for key in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            graphs[key].replay()
            end.record()
            end.synchronize()
            timings[key].append(begin.elapsed_time(end) * 1000 / args.buffers)
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "buffers": args.buffers,
        "input_bytes": args.buffers * 4096 * 256 * 4,
        "shape": [4096, 256],
        "dtype": "float32",
        "samples": args.samples,
        "bitwise_parity": True,
        "median_us": {
            name: {
                phase: statistics.median(timings[name, phase])
                for phase in ("forward", "backward", "forward_backward")
            }
            for name in ("native", "fused")
        },
        "samples_us": {
            f"{name}/{phase}": values for (name, phase), values in timings.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
