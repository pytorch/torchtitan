from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_FILES = sorted(ROOT.glob("llama3_8B_rank*.py"))

IMPORT_SNIPPET = """import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _helper_dir in (_THIS_DIR.parent / "candidate_models", _THIS_DIR):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))

from rope_extension import rope_backward_pair, rope_forward_pair
"""

RE_ASSIGN = re.compile(r"^(?P<indent>\s*)(?P<lhs>[^=]+?) = (?P<rhs>.+)$")
RE_TO_COPY = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\._to_copy\.default\((?P<input>\w+), dtype = torch\.float32.*$"
)
RE_VIEW = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.view\.default\((?P<input>\w+), \[1, 8192, (?P<heads>\d+), .+\]\);  (?P=input) = None$"
)
RE_VIEW_AS_COMPLEX = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.view_as_complex\.default\((?P<input>\w+)\);  (?P=input) = None$"
)
RE_SLICE = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.slice\.Tensor\((?P<input>\w+), 0, 0, 8192\)(?P<suffix>.*)$"
)
RE_FREQ_VIEW = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.view\.default\((?P<input>\w+), \[1, 8192, 1, 64\]\);  (?P=input) = None$"
)
RE_MUL = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.mul\.Tensor\((?P<input>\w+), (?P<freqs>\w+)\); .*$"
)
RE_VIEW_AS_REAL = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.view_as_real\.default\((?P<input>\w+)\);  (?P=input) = None$"
)
RE_BF16_CAST = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\._to_copy\.default\((?P<input>\w+), dtype = torch\.bfloat16.*$"
)
RE_CONJ = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\._conj\.default\((?P<input>\w+)\)(?P<suffix>.*)$"
)
RE_CLONE = re.compile(
    r"^(?P<indent>\s*)(?P<lhs>\w+): .+ = torch\.ops\.aten\.clone\.default\((?P<input>\w+)\);  (?P=input) = None$"
)


def _comment(line: str) -> str:
    if not line.strip():
        return line
    indent = line[: len(line) - len(line.lstrip(" "))]
    body = line[len(indent):]
    if body.startswith("# "):
        return line
    return f"{indent}# {body}"


def _lhs_prefix(line: str) -> str:
    match = RE_ASSIGN.match(line)
    if match is None:
        raise ValueError(f"unable to parse assignment line: {line!r}")
    return f"{match.group('indent')}{match.group('lhs').strip()}"


def _try_forward(lines: list[str], index: int) -> tuple[list[str], int] | None:
    if index + 16 >= len(lines):
        return None
    if "apply_rotary_emb_complex(" not in lines[index]:
        return None

    block = lines[index + 1 : index + 17]
    if (
        RE_TO_COPY.match(block[0]) is None
        or RE_VIEW.match(block[1]) is None
        or RE_VIEW_AS_COMPLEX.match(block[2]) is None
        or RE_TO_COPY.match(block[3]) is None
        or RE_VIEW.match(block[4]) is None
        or RE_VIEW_AS_COMPLEX.match(block[5]) is None
        or RE_SLICE.match(block[6]) is None
        or RE_FREQ_VIEW.match(block[7]) is None
        or RE_MUL.match(block[8]) is None
        or RE_VIEW_AS_REAL.match(block[9]) is None
        or RE_ASSIGN.match(block[10]) is None
        or RE_MUL.match(block[11]) is None
        or RE_VIEW_AS_REAL.match(block[12]) is None
        or RE_ASSIGN.match(block[13]) is None
        or RE_BF16_CAST.match(block[14]) is None
        or RE_BF16_CAST.match(block[15]) is None
    ):
        return None

    first_to_copy = RE_TO_COPY.match(block[0])
    second_to_copy = RE_TO_COPY.match(block[3])
    freqs_view = RE_FREQ_VIEW.match(block[7])

    indent = first_to_copy.group("indent")
    q_input = first_to_copy.group("input")
    k_input = second_to_copy.group("input")
    freqs_input = freqs_view.group("lhs")
    q_output = _lhs_prefix(block[14])
    k_output = _lhs_prefix(block[15])

    replacement = [lines[index]]
    replacement.extend(_comment(line) for line in block[0:6])
    replacement.extend(block[6:8])
    replacement.extend(_comment(line) for line in block[8:16])
    replacement.extend(
        [
            f"{indent}_rope_forward = rope_forward_pair({q_input}, {k_input}, {freqs_input})\n",
            f"{q_output} = _rope_forward[0]\n",
            f"{k_output} = _rope_forward[1];  _rope_forward = {q_input} = {k_input} = None\n",
        ]
    )
    return replacement, index + 17


def _try_backward(lines: list[str], index: int) -> tuple[list[str], int] | None:
    if index + 18 >= len(lines):
        return None
    if "apply_rotary_emb_complex(" not in lines[index]:
        return None

    block = lines[index + 1 : index + 19]
    if (
        RE_TO_COPY.match(block[0]) is None
        or RE_TO_COPY.match(block[1]) is None
        or RE_VIEW.match(block[2]) is None
        or RE_VIEW_AS_COMPLEX.match(block[3]) is None
        or RE_CONJ.match(block[4]) is None
        or RE_CLONE.match(block[5]) is None
        or RE_MUL.match(block[6]) is None
        or RE_VIEW.match(block[7]) is None
        or RE_VIEW_AS_COMPLEX.match(block[8]) is None
        or RE_CONJ.match(block[9]) is None
        or RE_CLONE.match(block[10]) is None
        or RE_MUL.match(block[11]) is None
        or RE_VIEW_AS_REAL.match(block[12]) is None
        or RE_ASSIGN.match(block[13]) is None
        or RE_BF16_CAST.match(block[14]) is None
        or RE_VIEW_AS_REAL.match(block[15]) is None
        or RE_ASSIGN.match(block[16]) is None
        or RE_BF16_CAST.match(block[17]) is None
    ):
        return None

    first_to_copy = RE_TO_COPY.match(block[0])
    second_to_copy = RE_TO_COPY.match(block[1])
    first_conj = RE_CONJ.match(block[4])
    second_conj = RE_CONJ.match(block[9])

    if first_conj.group("input") != second_conj.group("input"):
        raise ValueError("backward freqs cache mismatch")

    indent = first_to_copy.group("indent")
    k_input = first_to_copy.group("input")
    q_input = second_to_copy.group("input")
    freqs_input = first_conj.group("input")
    k_output = _lhs_prefix(block[14])
    q_output = _lhs_prefix(block[17])

    replacement = [lines[index]]
    replacement.extend(_comment(line) for line in block)
    replacement.extend(
        [
            f"{indent}_rope_backward = rope_backward_pair({k_input}, {q_input}, {freqs_input})\n",
            f"{k_output} = _rope_backward[0]\n",
            f"{q_output} = _rope_backward[1];  _rope_backward = {k_input} = {q_input} = {freqs_input} = None\n",
        ]
    )
    return replacement, index + 19


def transform_file(path: Path) -> tuple[int, int]:
    lines = path.read_text().splitlines(keepends=True)
    output: list[str] = []
    forward_count = 0
    backward_count = 0
    import_inserted = False

    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            not import_inserted
            and line == "from torch import device, tensor\n"
            and "from rope_extension import rope_backward_pair, rope_forward_pair\n"
            not in "".join(lines)
        ):
            output.append(line)
            output.append("\n")
            output.append(IMPORT_SNIPPET)
            import_inserted = True
            index += 1
            continue

        if "apply_rotary_emb_complex(" in line:
            forward_rewrite = _try_forward(lines, index)
            if forward_rewrite is not None:
                replacement, index = forward_rewrite
                output.extend(replacement)
                forward_count += 1
                continue

            backward_rewrite = _try_backward(lines, index)
            if backward_rewrite is not None:
                replacement, index = backward_rewrite
                output.extend(replacement)
                backward_count += 1
                continue

        output.append(line)
        index += 1

    if not import_inserted and "from rope_extension import rope_backward_pair, rope_forward_pair\n" not in "".join(lines):
        raise ValueError(f"failed to inject rope imports into {path}")
    if forward_count != 32:
        raise ValueError(f"expected 32 forward rewrites in {path}, got {forward_count}")
    if backward_count != 32:
        raise ValueError(f"expected 32 backward rewrites in {path}, got {backward_count}")

    path.write_text("".join(output))
    return forward_count, backward_count


def main() -> None:
    for path in TARGET_FILES:
        forward_count, backward_count = transform_file(path)
        print(f"{path.name}: forward={forward_count} backward={backward_count}")


if __name__ == "__main__":
    main()
