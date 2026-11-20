# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check a kernel against a kernel problem, bit for bit.

    python check_kernel.py <problem dir> <kernel.py> [--seeds N]

A problem directory (written by ``kernel_problems.py``) holds ``problem.py``
(``reference``), ``spec.json`` (the contract, and the layout of the inputs of each
case: calls of the training run) and ``inputs.pt`` (each case's integer, bool and
complex inputs, e.g. indices). ``kernel.py`` defines ``kernel_function`` with the
signature of ``reference``. For each case, with floating-point inputs generated
from each of ``--seeds`` seeds, the reference and the kernel run on their own
copies of the inputs, and

- each output must equal the reference's bit for bit, with the same shape, stride,
  dtype and device, and alias the same inputs and outputs;
- each input must afterwards equal the reference's copy bit for bit: the kernel
  makes the reference's in-place writes and no others.

Prints ``PASS`` and exits 0, or prints the first difference and exits 1. The file
only needs torch: ``kernel_problems.py`` copies it into each problem set. To use it
as a test that runs next to ``kernel.py``::

    import check_kernel
    check_kernel.main(["<problem dir>", "kernel.py"])
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

# Copies keep each tensor's address modulo this many bytes, so a kernel sees the
# alignment it sees in training (allocations are aligned to at least this).
_ALIGNMENT = 256
_SEEDS = 2


def load_module(path: str | Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    # Leave no __pycache__ in the problem directory.
    dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    return module


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """The bytes of its storage that ``t`` covers, ``[start, stop)``."""
    start = t.storage_offset() * t.element_size()
    if t.numel() == 0:
        return start, start
    extent = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return start, start + extent * t.element_size()


def _storage_spans(args: list[Any]) -> dict[tuple[torch.device, int], tuple[int, int]]:
    """Per storage that tensors of ``args`` view, the bytes they cover, from an
    address aligned to ``_ALIGNMENT``."""
    spans: dict[tuple[torch.device, int], tuple[int, int]] = {}
    for a in args:
        if isinstance(a, torch.Tensor):
            key = (a.device, a.untyped_storage().data_ptr())
            start, stop = _byte_span(a)
            if key in spans:
                start, stop = min(start, spans[key][0]), max(stop, spans[key][1])
            spans[key] = (start - start % _ALIGNMENT, stop)
    return spans


def describe_args(
    args: list[Any], *, generate_floating: bool, device: torch.device | str | None
) -> tuple[dict, list[torch.Tensor | None]]:
    """The layout of ``args`` and the bytes of their storages.

    The layout (JSON) lists the storages that the tensors view (``nbytes``, and
    whether their values are generated) and per input its storage, byte offset,
    shape, stride and dtype, or its value. The bytes are a copy of each storage on
    ``device`` (default: where it is), or None for a storage whose values are
    generated: with ``generate_floating``, a storage that only floating-point
    tensors view. ``build_args`` makes inputs from the two.
    """
    spans = _storage_spans(args)
    index = {key: i for i, key in enumerate(spans)}
    views: dict[tuple[torch.device, int], list[torch.Tensor]] = {k: [] for k in spans}
    for a in args:
        if isinstance(a, torch.Tensor):
            views[(a.device, a.untyped_storage().data_ptr())].append(a)
    storages, data = [], []
    for key, (start, stop) in spans.items():
        generated = generate_floating and all(t.is_floating_point() for t in views[key])
        storages.append({"nbytes": stop - start, "generated": generated})
        if generated:
            data.append(None)
            continue
        src = torch.empty(0, dtype=torch.uint8, device=key[0])
        src.set_(views[key][0].untyped_storage(), start, (stop - start,), (1,))
        data.append(src.to(device or key[0], copy=True))
    inputs = []
    for a in args:
        if isinstance(a, torch.Tensor):
            key = (a.device, a.untyped_storage().data_ptr())
            inputs.append(
                {
                    "shape": list(a.shape),
                    "stride": list(a.stride()),
                    "dtype": str(a.dtype).removeprefix("torch."),
                    "storage": index[key],
                    "offset": _byte_span(a)[0] - spans[key][0],
                }
            )
        else:
            inputs.append({"value": a})
    return {"storages": storages, "inputs": inputs}, data


def build_args(
    layout: dict, data: list[torch.Tensor | None], seed: int, device: str = "cuda"
) -> list[Any]:
    """Inputs as ``describe_args`` described them: views of each storage's bytes,
    or of a new storage on ``device`` whose floating-point views are filled with
    random values from ``seed``."""
    buffers = [
        torch.empty(s["nbytes"], dtype=torch.uint8, device=device) if d is None else d
        for s, d in zip(layout["storages"], data)
    ]
    generator = None
    args = []
    for described in layout["inputs"]:
        if "shape" not in described:
            args.append(described["value"])
            continue
        dtype = getattr(torch, described["dtype"])
        buffer = buffers[described["storage"]]
        t = torch.empty(0, dtype=dtype, device=buffer.device)
        t.set_(
            buffer.untyped_storage(),
            described["offset"] // t.element_size(),
            described["shape"],
            described["stride"],
        )
        if data[described["storage"]] is None and t.numel():
            # Fill the bytes the view covers (a view may repeat elements).
            start, stop = _byte_span(t)
            flat = torch.empty(0, dtype=dtype, device=buffer.device)
            n = (stop - start) // t.element_size()
            flat.set_(buffer.untyped_storage(), t.storage_offset(), (n,), (1,))
            if generator is None:
                generator = torch.Generator(device=device).manual_seed(seed)
            values = torch.randn(n, generator=generator, device=device)
            if dtype == torch.float8_e8m0fnu:  # unsigned: scales
                values = values.abs()
            flat.copy_(values.to(dtype))
        args.append(t)
    return args


def copy_args(args: list[Any], device: torch.device | str | None = None) -> list[Any]:
    """Copies of ``args`` (on ``device``, or where each tensor is). A tensor keeps its
    shape, stride, dtype and address modulo ``_ALIGNMENT``; tensors that share a
    storage share one copy of the bytes they cover, so in-place writes through one
    are seen through the others. Other values are kept."""
    layout, data = describe_args(args, generate_floating=False, device=device)
    return build_args(layout, data, seed=0)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def _aliasing(inputs: list[Any], outputs: tuple) -> list[list[str]]:
    """Per output, the inputs (``in<k>``) and earlier outputs (``out<k>``) whose
    storage it shares."""
    storages = [
        (f"in{k}", a.untyped_storage().data_ptr())
        for k, a in enumerate(inputs)
        if isinstance(a, torch.Tensor)
    ]
    result = []
    for k, o in enumerate(outputs):
        ptr = o.untyped_storage().data_ptr() if isinstance(o, torch.Tensor) else None
        result.append([name for name, p in storages if p == ptr])
        storages.append((f"out{k}", ptr))
    return result


def compare(
    expected_outputs: Any,
    outputs: Any,
    expected_inputs: list[Any],
    inputs: list[Any],
) -> str | None:
    """The first difference between two runs of a problem on equal copies of its
    inputs (outputs, and the inputs after the run), or None."""
    if not isinstance(outputs, tuple) or len(outputs) != len(expected_outputs):
        return (
            f"returns {type(outputs).__name__} of length "
            f"{len(outputs) if isinstance(outputs, (tuple, list)) else '-'}, "
            f"expected a tuple of {len(expected_outputs)}"
        )
    pairs = [
        (f"output {k}", e, o) for k, (e, o) in enumerate(zip(expected_outputs, outputs))
    ]
    pairs += [
        (f"input {k} after the call", e, o)
        for k, (e, o) in enumerate(zip(expected_inputs, inputs))
    ]
    for what, e, o in pairs:
        if not isinstance(e, torch.Tensor):
            if e != o:
                return f"{what}: {o!r}, expected {e!r}"
            continue
        if not isinstance(o, torch.Tensor):
            return f"{what}: {type(o).__name__}, expected a tensor"
        for attr in ("shape", "dtype"):
            if getattr(o, attr) != getattr(e, attr):
                return f"{what}: {attr} {getattr(o, attr)}, expected {getattr(e, attr)}"
        if o.stride() != e.stride():
            return f"{what}: stride {o.stride()}, expected {e.stride()}"
        if o.device.type != e.device.type:
            return f"{what}: device {o.device}, expected {e.device}"
        if not _same_bits(o, e):
            num = int(
                (
                    o.reshape(-1).view(torch.uint8) != e.reshape(-1).view(torch.uint8)
                ).sum()
            )
            return f"{what}: {num} bytes differ"
    got, want = _aliasing(inputs, outputs), _aliasing(expected_inputs, expected_outputs)
    if got != want:
        return f"aliasing: outputs share storage with {got}, expected {want}"
    return None


def load_cases(
    problem_dir: str | Path, seed: int = 0, device: str = "cuda"
) -> list[list[Any]]:
    """The inputs of each case of a problem: the integer, bool and complex inputs
    saved from the training run, and floating-point inputs generated from
    ``seed``."""
    problem_dir = Path(problem_dir)
    spec = json.loads((problem_dir / "spec.json").read_text())
    saved = torch.load(
        problem_dir / "inputs.pt", map_location=device, weights_only=True, mmap=True
    )
    return [
        build_args(case, data, seed, device) for case, data in zip(spec["cases"], saved)
    ]


def check(
    problem_dir: str | Path, kernel_path: str | Path, seeds: int = _SEEDS
) -> str | None:
    """The first difference between the kernel and the reference, or None."""
    problem_dir = Path(problem_dir)
    spec = json.loads((problem_dir / "spec.json").read_text())
    torch.use_deterministic_algorithms(spec["env"]["deterministic"])
    reference = load_module(problem_dir / "problem.py", "problem").reference
    kernel_function = load_module(kernel_path, "kernel").kernel_function
    if not spec["cases"]:
        return "no input cases"
    for seed in range(seeds):
        for k, case in enumerate(load_cases(problem_dir, seed)):
            ref_inputs, inputs = copy_args(case), copy_args(case)
            expected = reference(*ref_inputs)
            try:
                outputs = kernel_function(*inputs)
            except Exception as e:  # noqa: BLE001 - reported as the difference
                return f"case {k} (seed {seed}): kernel_function raised {e!r}"
            torch.accelerator.synchronize()
            difference = compare(expected, outputs, ref_inputs, inputs)
            if difference is not None:
                call = spec["cases"][k]["call"]
                return f"case {k} ({call}, seed {seed}): {difference}"
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("problem_dir")
    parser.add_argument("kernel")
    parser.add_argument("--seeds", type=int, default=_SEEDS)
    args = parser.parse_args(argv)
    difference = check(args.problem_dir, args.kernel, args.seeds)
    if difference is None:
        print("PASS")
        sys.exit(0)
    print(f"FAIL: {difference}")
    sys.exit(1)


if __name__ == "__main__":
    main()
