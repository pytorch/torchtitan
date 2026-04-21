from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_FILES = sorted(ROOT.glob("llama3_8B_rank*.py"))

IMPORT_NEEDLE = "from rope_extension import rope_backward_pair, rope_forward_pair\n"
IMPORT_INSERT = (
    "from rope_extension import rope_backward_pair, rope_forward_pair\n"
    "from cudagraph_wrapper import maybe_wrap_forward\n"
)
FOOTER = "\nGraphModule.forward = maybe_wrap_forward(GraphModule.forward)\n"


def transform_file(path: Path) -> None:
    text = path.read_text()
    if "from cudagraph_wrapper import maybe_wrap_forward\n" not in text:
        text = text.replace(IMPORT_NEEDLE, IMPORT_INSERT, 1)
    if "GraphModule.forward = maybe_wrap_forward(GraphModule.forward)\n" not in text:
        text = text.rstrip() + FOOTER
    path.write_text(text)


def main() -> None:
    for path in TARGET_FILES:
        transform_file(path)
        print(path.name)


if __name__ == "__main__":
    main()
