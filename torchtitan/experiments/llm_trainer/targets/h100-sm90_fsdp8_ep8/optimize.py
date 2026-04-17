#!/usr/bin/env python3
"""
Optimize DSV3 candidate model files by removing redundant operations
and fusing elementwise kernels.

Applies these transformations (in order):
1. Remove dead code (operations whose result is immediately set to None)
2. Cache the causal attention mask (created 6x, reuse 1st)
3. Replace clone+_unsafe_view pairs with reshape (1 op instead of 2)
4. Add Triton SwiGLU fusion (silu+mul → 1 kernel)
5. Remove redundant no-op views (view to same shape)
"""

import re
import sys
from pathlib import Path


def remove_dead_code(source: str) -> str:
    """Remove operations whose result is immediately set to None on the same line.

    These are operations like:
        _to_copy_13: "bf16[...]" = torch.ops.aten._to_copy.default(...);  _to_copy_13 = None

    The result is never used - pure waste (especially large bf16 softmax copies).
    """
    lines = source.split("\n")
    result = []
    removed = 0

    # Pattern: varname: "type" = expr;  varname = None
    dead_pattern = re.compile(
        r'^(\s+)(\w+):\s+"[^"]*"\s+=\s+.*;\s+\2 = None\s*$'
    )

    for line in lines:
        m = dead_pattern.match(line)
        if m:
            indent = m.group(1)
            var = m.group(2)
            result.append(f"{indent}# REMOVED dead code: {var} (result immediately discarded)")
            removed += 1
        else:
            result.append(line)

    print(f"    Removed {removed} dead operations")
    return "\n".join(result)


def cache_causal_mask(source: str) -> str:
    """Replace repeated causal mask creation with a cached version.

    The pattern (appears 6 times, once per attention layer):
        ones_N = torch.ops.aten.ones.default([2048, 2048], dtype=bool, ...)
        tril_N = torch.ops.aten.tril.default(ones_N)
        scalar_tensor_2N = torch.ops.aten.scalar_tensor.default(-inf, ...)
        scalar_tensor_2N+1 = torch.ops.aten.scalar_tensor.default(0.0, ...)
        where_N = torch.ops.aten.where.self(tril_N, scalar_tensor_2N+1, scalar_tensor_2N)
    """
    # Find all causal mask blocks: ones → tril → scalar(-inf) → scalar(0) → where
    mask_pattern = re.compile(
        r"(        (\w+): \"b8\[2048, 2048\][^\"]*\" = torch\.ops\.aten\.ones\.default\(\[2048, 2048\][^\n]*\n)"
        r"(        (\w+): \"b8\[2048, 2048\][^\"]*\" = torch\.ops\.aten\.tril\.default\(\2\);[^\n]*\n)"
        r"(        (\w+): \"f32\[\]\[\][^\"]*\" = torch\.ops\.aten\.scalar_tensor\.default\(-inf[^\n]*\n)"
        r"(        (\w+): \"f32\[\]\[\][^\"]*\" = torch\.ops\.aten\.scalar_tensor\.default\(0\.0[^\n]*\n)"
        r"(        (\w+): \"f32\[2048, 2048\][^\"]*\" = torch\.ops\.aten\.where\.self\(\4, \8, \6\);[^\n]*\n)"
    )

    matches = list(mask_pattern.finditer(source))
    if len(matches) <= 1:
        print(f"    Found {len(matches)} causal mask blocks, nothing to cache")
        return source

    first_where_var = matches[0].group(10)
    replaced = 0

    # Replace subsequent occurrences (reverse order to preserve positions)
    for m in reversed(matches[1:]):
        where_var = m.group(10)
        old_text = m.group(0)
        new_text = f"        # CACHED: reuse causal mask from first attention layer\n        {where_var} = {first_where_var}\n"
        source = source.replace(old_text, new_text)
        replaced += 1

    print(f"    Cached causal mask: kept 1, replaced {replaced} duplicates")
    return source


def replace_clone_unsafe_view(source: str) -> str:
    """Replace clone(x, contiguous) + _unsafe_view(clone, shape) with reshape(x, shape).

    This saves one memory allocation and copy per occurrence.
    """
    # Match clone immediately followed by _unsafe_view using clone's output
    pattern = re.compile(
        r"(        (\w+): \"([^\"]*)\" = torch\.ops\.aten\.clone\.default\((\w+), memory_format = torch\.contiguous_format\))([^\n]*)\n"
        r"(        (\w+): \"([^\"]*)\" = torch\.ops\.aten\._unsafe_view\.default\(\2, (\[[^\]]+\])\))([^\n]*)"
    )

    count = 0
    def replacer(m):
        nonlocal count
        count += 1
        clone_line = m.group(1)
        input_var = m.group(4)
        clone_cleanup = m.group(5)
        view_var = m.group(7)
        view_type = m.group(8)
        shape = m.group(9)
        view_cleanup = m.group(10)

        return (
            f"        # {clone_line.strip()}\n"
            f"        {view_var}: \"{view_type}\" = torch.ops.aten.reshape.default({input_var}, {shape}){view_cleanup}"
        )

    source = pattern.sub(replacer, source)
    print(f"    Replaced {count} clone+_unsafe_view pairs with reshape")
    return source


def add_triton_swiglu(source: str) -> str:
    """Add Triton SwiGLU kernel and replace silu+mul patterns in forward pass.

    Forward pattern:
        silu_N = torch.ops.aten.silu.default(x)
        ...  (optional comments)
        mul_N = torch.ops.aten.mul.Tensor(silu_N, y)

    We only fuse forward SwiGLU, not backward silu_backward (different pattern).
    """
    triton_code = '''
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    X_ptr, Y_ptr, OUT_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    sigmoid_x = tl.sigmoid(x.to(tl.float32)).to(x.dtype)
    silu_x = x * sigmoid_x
    out = silu_x * y
    tl.store(OUT_ptr + offsets, out, mask=mask)


def fused_swiglu(x, y):
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _swiglu_fwd_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

'''

    # Insert after existing imports
    import_marker = "\nclass GraphModule"
    pos = source.find(import_marker)
    if pos == -1:
        return source
    source = source[:pos] + "\n" + triton_code + source[pos:]

    # Two-pass approach:
    # Pass A: Find all silu outputs and their inputs
    # Pass B: Find all mul(silu_var, other) and replace with fused call
    lines = source.split("\n")

    silu_re = re.compile(
        r'^(        )(\w+): "([^"]*)" = torch\.ops\.aten\.silu\.default\((\w+)\)(.*)'
    )
    mul_re = re.compile(
        r'^(        )(\w+): "([^"]*)" = torch\.ops\.aten\.mul\.Tensor\((\w+), (\w+)\)(.*)'
    )

    # Pass A: collect silu var → silu input mapping, and line indices
    silu_info = {}  # silu_var -> (silu_input, line_index, silu_type)
    for i, line in enumerate(lines):
        m = silu_re.match(line)
        if m:
            silu_var = m.group(2)
            silu_input = m.group(4)
            silu_type = m.group(3)
            silu_info[silu_var] = (silu_input, i, silu_type)

    # Pass B: find mul lines that reference a silu var
    comment_silu_lines = set()
    count = 0
    result_lines = []
    for i, line in enumerate(lines):
        if i in comment_silu_lines:
            # Comment out this silu line
            result_lines.append("        # " + line.strip())
            continue

        m = mul_re.match(line)
        if m:
            indent = m.group(1)
            mul_var = m.group(2)
            mul_type = m.group(3)
            mul_arg1 = m.group(4)
            mul_arg2 = m.group(5)
            mul_rest = m.group(6)

            if mul_arg1 in silu_info:
                silu_input, silu_line_idx, silu_type = silu_info[mul_arg1]
                # Comment out original mul and add fused call
                result_lines.append(
                    f'{indent}# {mul_var}: "{mul_type}" = torch.ops.aten.mul.Tensor({mul_arg1}, {mul_arg2})'
                )
                result_lines.append(
                    f'{indent}{mul_var}: "{mul_type}" = fused_swiglu({silu_input}, {mul_arg2}){mul_rest}'
                )
                comment_silu_lines.add(silu_line_idx)
                count += 1
                continue

        result_lines.append(line)

    # If we found patterns but silu lines are after the mul lines (shouldn't happen),
    # do a second pass to comment out silu lines we missed
    if comment_silu_lines:
        final_lines = []
        for i, line in enumerate(result_lines):
            # The indices refer to original lines, but we may have shifted.
            # Since we only add/modify lines at mul positions, silu line indices
            # should still match if they come before the first mul.
            final_lines.append(line)
        result_lines = final_lines

    print(f"    Fused {count} SwiGLU (silu+mul) patterns")
    return "\n".join(result_lines)


def remove_noop_views(source: str) -> str:
    """Remove view operations that don't change the shape.

    Pattern: var_out: "type[shape]..." = torch.ops.aten.view.default(var_in, [shape])
    where var_in has the same shape as the output.
    """
    lines = source.split("\n")

    # First pass: collect variable shapes from type annotations
    var_shapes = {}
    shape_re = re.compile(r'^\s+(\w+):\s+"[a-z0-9]+\[([^\]]+)\]')
    for line in lines:
        m = shape_re.match(line)
        if m:
            var_shapes[m.group(1)] = m.group(2).replace(" ", "")

    # Second pass: find no-op views and build alias map
    alias_map = {}
    view_re = re.compile(
        r'^\s+(\w+):\s+"[^"]+"\s+=\s+torch\.ops\.aten\.view\.default\((\w+),\s+\[([^\]]+)\]\)'
    )
    for line in lines:
        m = view_re.match(line)
        if m:
            out_var, in_var, target = m.group(1), m.group(2), m.group(3).replace(" ", "")
            # Resolve input through alias chain
            resolved = in_var
            while resolved in alias_map:
                resolved = alias_map[resolved]
            if resolved in var_shapes:
                if var_shapes[resolved] == target:
                    alias_map[out_var] = in_var
            elif in_var in var_shapes and out_var in var_shapes:
                if var_shapes[in_var] == var_shapes[out_var]:
                    alias_map[out_var] = in_var

    if not alias_map:
        print("    No redundant views found")
        return source

    # Resolve full alias chains
    def resolve(v):
        seen = set()
        while v in alias_map and v not in seen:
            seen.add(v)
            v = alias_map[v]
        return v

    # Build set of lines that define aliased variables
    remove_lines = set()
    assign_re = re.compile(r'^(\s+)(\w+)\s*(?::\s*"[^"]*")?\s*=')
    for i, line in enumerate(lines):
        m = assign_re.match(line)
        if m and m.group(2) in alias_map:
            remove_lines.add(i)

    # Third pass: remove aliased lines, substitute references in remaining
    result = []
    for i, line in enumerate(lines):
        if i in remove_lines:
            continue
        new_line = line
        for old_var in alias_map:
            target = resolve(old_var)
            if old_var != target:
                # Negative lookbehind for '.' avoids replacing method names
                # like torch.ops.aten.view.default when var is named "view"
                new_line = re.sub(r'(?<!\.)' + r'\b' + re.escape(old_var) + r'\b', target, new_line)
        result.append(new_line)

    print(f"    Removed {len(remove_lines)} redundant view operations")
    return "\n".join(result)


def remove_noop_slices(source: str) -> str:
    """Remove slice operations that select the full tensor.

    Pattern: out = torch.ops.aten.slice.Tensor(x, 0, 0, N)
    where x.shape[0] == N (so the slice is a no-op).
    """
    lines = source.split("\n")

    # Collect shapes
    var_shapes = {}
    shape_re = re.compile(r'^\s+(\w+):\s+"[a-z0-9]+\[([^\]]+)\]')
    for line in lines:
        m = shape_re.match(line)
        if m:
            var_shapes[m.group(1)] = m.group(2)

    # Find no-op slices
    alias_map = {}
    slice_re = re.compile(
        r'^\s+(\w+):\s+"[^"]+"\s+=\s+torch\.ops\.aten\.slice\.Tensor\((\w+),\s+0,\s+0,\s+(\d+)\)'
    )
    for line in lines:
        m = slice_re.match(line)
        if m:
            out_var, in_var, end = m.group(1), m.group(2), int(m.group(3))
            resolved = in_var
            while resolved in alias_map:
                resolved = alias_map[resolved]
            if resolved in var_shapes:
                first_dim = var_shapes[resolved].split(",")[0].strip()
                try:
                    if int(first_dim) == end:
                        alias_map[out_var] = in_var
                except ValueError:
                    pass

    if not alias_map:
        print("    No redundant slices found")
        return source

    def resolve(v):
        seen = set()
        while v in alias_map and v not in seen:
            seen.add(v)
            v = alias_map[v]
        return v

    remove_lines = set()
    assign_re = re.compile(r'^(\s+)(\w+)\s*(?::\s*"[^"]*")?\s*=')
    for i, line in enumerate(lines):
        m = assign_re.match(line)
        if m and m.group(2) in alias_map:
            remove_lines.add(i)

    result = []
    for i, line in enumerate(lines):
        if i in remove_lines:
            continue
        new_line = line
        for old_var in alias_map:
            target = resolve(old_var)
            if old_var != target:
                new_line = re.sub(r'(?<!\.)' + r'\b' + re.escape(old_var) + r'\b', target, new_line)
        result.append(new_line)

    print(f"    Removed {len(remove_lines)} redundant slice operations")
    return "\n".join(result)


def optimize_file(filepath: Path) -> str:
    """Apply all optimizations to a model file."""
    source = filepath.read_text()
    orig_lines = len(source.splitlines())
    print(f"  Original: {orig_lines} lines")

    source = remove_dead_code(source)
    source = cache_causal_mask(source)
    source = replace_clone_unsafe_view(source)
    source = add_triton_swiglu(source)
    # NOTE: remove_noop_slices and remove_noop_views are disabled because
    # they break variable lifetime (cleanup "; var = None" on removed lines
    # is lost, causing downstream uses to find None). Need more careful
    # handling of cleanup statements before re-enabling.
    # source = remove_noop_slices(source)
    # source = remove_noop_views(source)

    final_lines = len(source.splitlines())
    print(f"  Final: {final_lines} lines (removed {orig_lines - final_lines})")
    return source


def main():
    target_dir = Path(__file__).parent / "candidate_models"
    model_files = sorted(target_dir.glob("deepseek_v3_debugmodel_rank*.py"))

    if not model_files:
        print("No candidate model files found!")
        sys.exit(1)

    print(f"Found {len(model_files)} model files to optimize\n")

    for filepath in model_files:
        print(f"Optimizing {filepath.name}...")
        optimized = optimize_file(filepath)
        filepath.write_text(optimized)
        print()

    print("Done! Run the benchmarker to verify bitwise equivalence.")


if __name__ == "__main__":
    main()
