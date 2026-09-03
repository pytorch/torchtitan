---
name: kernel-numerics-verifier
description: Verify forward and backward numerical correctness of a handwritten low-precision GPU kernel against a high-precision oracle and an eager same-dtype baseline. Use for Triton or CUDA kernel correctness; use numerics_debugging to locate an unknown whole-model divergence.
---

# Kernel Numerics Verifier

Verify a handwritten kernel with three implementations:

- `golden`: eager FP64 implementation. Use FP32 only when FP64 is unsupported.
- `reference`: eager implementation in the kernel's input and output dtype.
- `target`: handwritten Triton or CUDA kernel in the same dtype as `reference`.

Generate low-precision inputs once, then cast those exact values to FP64 for
`golden`. This measures kernel arithmetic error without mixing in input
quantization error.

## Correctness gate

For each output tensor and gradient tensor, compute in the golden dtype:

```text
reference_error = max(abs(reference - golden))
target_error = max(abs(target - golden))
rounding_floor = max(abs(golden.to(target_dtype).to(golden_dtype) - golden))
absolute_floor = max(project_atol, rounding_floor)
threshold = fudge_factor * reference_error + absolute_floor
```

When the reference is sufficiently accurate, pass when:

```text
target_error <= threshold
```

Use an existing operator tolerance when one exists. Otherwise start with
`fudge_factor = 2.0` and `project_atol = 0`. Declare them before running the
target. Do not increase them merely to make a failing kernel pass.

## Check conditioning before gating

Measure whether the reference error is material relative to the golden result:

```text
golden_scale = max(abs(golden))
reference_relative_error =
    reference_error / max(golden_scale, absolute_floor)
```

Predeclare an acceptable reference-relative error. If none exists, use 10% as
a diagnostic trigger. When the reference exceeds it, do not issue a pass or
fail from the 2x gate; the permitted error is too large to distinguish a bug
from numerical sensitivity.

For a reduction, inspect the output element responsible for the error and
capture its FP64 per-term contributions:

```text
condition_number = sum(abs(per_term)) / abs(sum(per_term))
```

A large condition number means cancellation amplifies small errors in the
terms. It does not by itself prove that the target is correct.

When the gate is invalid:

1. Compare two legitimate FP32 reduction orders. This measures sensitivity to
   accumulation order.
2. Repeat across several reduction lengths and seeds. Error growing roughly
   with `sqrt(T)` indicates zero-mean numerical noise; growth with `T` indicates
   systematic bias. Track the condition number at each `T`.
3. Compare reference and target error distributions and bias. A target-specific
   bias, abnormal scaling, or semantic mismatch is a failure.

If both implementations follow the same conditioning-driven noise envelope,
report `PASS WITH LIMITATIONS`. If the evidence cannot distinguish numerical
noise from a kernel bug, report `INCONCLUSIVE`.

For simple elementwise kernels, also require the result to be correctly rounded
or within 1 to 2 ULPs. Larger factors require evidence from an already accepted
implementation across representative inputs.

## Workflow

1. Read the kernel, eager formula, callsites, and existing tests. Record the
   formula, intermediate dtypes, accumulator dtypes, output dtype, supported
   shapes, masking, and NaN/Inf behavior.
2. Create identical low-precision inputs for `reference` and `target`. Create
   `golden` inputs by losslessly upcasting those values.
3. Compare shape, dtype, NaN locations, and positive and negative Inf locations
   before applying numerical tolerances. Any mismatch is a failure.
4. Check the reference-relative error. Apply the correctness gate only when the
   reference is sufficiently accurate; otherwise run the conditioning analysis.
5. Apply the resulting decision procedure to every forward output.
6. Use one random upstream gradient for all three implementations. Check every
   input and parameter gradient separately.
7. Test production shapes, the smallest legal shape, and dimensions immediately
   below and above relevant kernel tile sizes. Add adversarial values relevant
   to the operation, such as cancellation, extreme logits, or near-zero
   variance.
8. If the kernel is nondeterministic, repeat identical inputs and require every
   result to pass the same gate.
9. Diagnose failures before changing tolerances. Check indexing and masks first,
   then accumulator dtype, premature casts, reduction order, numerical
   stabilization, approximation functions, and backward formulas.

## Result

Report a compact table for forward and each gradient:

| Tensor | Ref/golden | Condition | Target error | Threshold | Gate valid | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |

Give one verdict: `PASS`, `PASS WITH LIMITATIONS`, `FAIL`, or `INCONCLUSIVE`.
State the tested shapes, dtypes, and unsupported cases. Do not claim general
correctness from one shape or forward-only testing.

Benchmark performance only after numerical verification passes.
