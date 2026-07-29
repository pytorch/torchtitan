#!/usr/bin/env python3
# Temporary CI probe for https://github.com/pytorch/pytorch/pull/190402.
#
# This rewrites the installed PyTorch nightly back to the scheduler/inversion
# behavior before PR 190402. It is intentionally narrow and should be removed
# once causality is proven.

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path


OLD_SHARED_DATA_AFTER_INVERTING_INDEXING = '''    def shared_data_after_inverting_indexing(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        """
        Attempts to enable fusion between two nodes by inverting indexing patterns.

        This optimization targets cases where node1 has a contiguous write and
        node2 has a contiguous write but discontiguous read. By inverting the
        indexing in node2's read and write operations, we can make them compatible
        with node1 for potential fusion.

        Args:
            node1: First scheduler node (source)
            node2: Second scheduler node (target for inversion)

        Returns:
            int: Fusion score if successful, 0 if optimization not applicable
        """

        if not config.loop_index_inversion_in_fusion:
            return -1

        if any(n.is_cpu() for n in [node1, node2]):
            return -1
        if not isinstance(node2, SchedulerNode):
            return -1
        if not isinstance(node2.node, ir.ComputedBuffer):
            return -1
        body = node2._body
        if body is None:
            return -1

        # Check for shared buffers between nodes
        node1_buffer_names = node1.read_writes.buffer_names()
        node2_buffer_names = node2.read_writes.buffer_names()
        common_buffer_names = node1_buffer_names & node2_buffer_names

        if not common_buffer_names:
            return -1

        # only invert if node1 is single unmet dep
        node2_unmet_dependencies = OrderedSet(
            dep.name for dep in node2.unmet_dependencies
        )
        if node2_unmet_dependencies - node1_buffer_names:
            return -1

        if len(node2_unmet_dependencies) > 1:
            return -1

        # Currently only handle single read/write operations
        if len(node2.read_writes.reads) != 1 or len(node2.read_writes.writes) != 1:
            return -1

        node2_read = next(iter(node2.read_writes.reads))
        node2_write = next(iter(node2.read_writes.writes))

        if not isinstance(node2_read, MemoryDep) or not isinstance(
            node2_write, MemoryDep
        ):
            return -1

        matching_node1_writes = [
            dep for dep in node1.read_writes.writes if dep.name == node2_read.name
        ]
        if len(matching_node1_writes) != 1:
            return -1

        node1_write = matching_node1_writes[0]

        if not isinstance(node1_write, MemoryDep):
            return -1

        # We are checking for compatibility with the normalized node1 write
        # then modifying node2 reads/writes. since the node1 write will be just used
        # for compatibility, while node2 will be used in actual modification, just
        # normalize node1 not node2.
        node1_write = node1_write.normalize()

        if (
            node1_write.index != node2_write.index
            and node1_write.size != node2_write.size
        ):
            return -1

        if node2_read.size != node2_write.size or len(node2_read.var_names) != 1:
            return -1

        # Verify we have exactly two indexing expressions (one read, one write)
        if len(body.indexing_exprs) != 2:
            return -1

        # No subblocks allowed for this optimization
        if body.subblocks:
            return -1

        if not ("index0" in body.indexing_exprs and "index1" in body.indexing_exprs):
            raise AssertionError("expected index0 and index1 in node2 indexing_exprs")

        # Extract and verify single read expression
        node2_read_exprs = OrderedSet(body.get_read_exprs())
        if len(node2_read_exprs) != 1:
            return -1

        read_expr = next(iter(node2_read_exprs))

        # Determine which index is for reading vs writing
        if read_expr == body.indexing_exprs["index0"]:
            read_expr_index = "index0"
            write_expr_index = "index1"
        else:
            if read_expr != body.indexing_exprs["index1"]:
                raise AssertionError("expected read_expr to match node2 index1 expr")
            read_expr_index = "index1"
            write_expr_index = "index0"

        from torch._inductor.invert_expr_analysis import generate_inverse_formula

        index_vars = body.vars[0]
        if len(index_vars) != 1:
            return -1

        simplified_terms = []
        for term in sympy.Add.make_args(read_expr):
            simplified_terms.append(
                V.graph.sizevars.combine_modular_indexing_pairs(term)
            )
        simplified_read_expr = sum(simplified_terms)

        inverse_formula = generate_inverse_formula(
            simplified_read_expr, index_vars[0], node2_read.size[0]
        )

        # formula is not invertible
        if inverse_formula is None:
            return -1

        # === Apply Inversion ===

        # Swap the indexing expressions using the inverse formula
        node2.apply_indexing_exprs(
            {
                read_expr_index: body.indexing_exprs[write_expr_index],
                write_expr_index: inverse_formula,
            }
        )

        # Calculate fusion score
        score = self.score_fusion_memory(node1, node2)
        if not isinstance(score, int):
            raise AssertionError("expected score to be an int")
        if score == 0:
            score = self._score_fusion_memory_by_fusable_read_write(node1, node2)

        fusion_log.info("Shared memory after inversion: %d", score)
        return score

'''


INVERT_EXPR_190402_BLOCK = '''    # Collapse modular fragments introduced by view decomposition before parsing.
    expr = sympy.Add(
        *(
            V.graph.sizevars.combine_modular_indexing_pairs(term)
            for term in sympy.Add.make_args(expr)
        )
    )
    expr = expr.replace(lambda subexpr: isinstance(subexpr, sympy.Add), join_dimensions)

'''


def get_torch_dir() -> Path:
    override = os.environ.get("TORCH_PACKAGE_DIR")
    if override:
        return Path(override).resolve()

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate installed torch package")
    return Path(spec.origin).resolve().parent


def read_optional_str_assignment(text: str, name: str) -> str | None:
    match = re.search(rf"^{name}: Optional\[str\] = (.+)$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not find {name} in torch/version.py")
    value = match.group(1)
    if value == "None":
        return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    raise RuntimeError(f"Unexpected {name} value in torch/version.py: {value}")


def get_cuda_and_hip_versions(torch_dir: Path) -> tuple[str | None, str | None]:
    version_text = (torch_dir / "version.py").read_text()
    return (
        read_optional_str_assignment(version_text, "cuda"),
        read_optional_str_assignment(version_text, "hip"),
    )


def write_if_changed(path: Path, old_text: str, new_text: str) -> bool:
    if old_text == new_text:
        return False
    path.write_text(new_text)
    print(f"patched {path}")
    return True


def patch_invert_expr_analysis(path: Path) -> bool:
    text = path.read_text()
    updated = text.replace("from .sizevars import join_dimensions\n", "", 1)
    updated = updated.replace(INVERT_EXPR_190402_BLOCK, "", 1)

    if "join_dimensions" in updated:
        raise RuntimeError(f"{path} still references join_dimensions after patch")
    if "Collapse modular fragments introduced by view decomposition" in updated:
        raise RuntimeError(f"{path} still contains PR 190402 modular-collapse block")

    return write_if_changed(path, text, updated)


def remove_scheduler_helper_block(text: str) -> str:
    helper = "    def _can_reindex_consumer_for_index_inversion("
    target = "    def shared_data_after_inverting_indexing("
    if helper not in text:
        return text

    start = text.index(helper)
    end = text.index(target, start)
    return text[:start] + text[end:]


def replace_scheduler_inversion_method(text: str) -> str:
    target = "    def shared_data_after_inverting_indexing("
    next_method = "    def shared_data_after_reordering_loop("
    start = text.index(target)
    end = text.index(next_method, start)
    return text[:start] + OLD_SHARED_DATA_AFTER_INVERTING_INDEXING + text[end:]


def patch_scheduler(path: Path) -> bool:
    text = path.read_text()
    updated = text.replace("    decompose_index,\n", "", 1)
    updated = updated.replace(
        '_FLATTENED_READ_VAR = sympy.Dummy("flattened_read", integer=True, nonnegative=True)\n',
        "",
        1,
    )
    updated = updated.replace(
        "        self._graph_partition_counter = itertools.count()\n"
        "        self.completed_operations: OrderedSet[str] = OrderedSet()\n",
        "        self._graph_partition_counter = itertools.count()\n\n"
        "        self.completed_operations: OrderedSet[str] = OrderedSet()\n",
        1,
    )
    updated = remove_scheduler_helper_block(updated)
    updated = replace_scheduler_inversion_method(updated)

    leftover_markers = [
        "decompose_index",
        "_FLATTENED_READ_VAR",
        "_can_reindex_consumer_for_index_inversion",
        "_get_flattened_read_inverse",
        "_get_canonical_indexing_inverse",
    ]
    leftovers = [marker for marker in leftover_markers if marker in updated]
    if leftovers:
        raise RuntimeError(f"{path} still contains PR 190402 markers: {leftovers}")

    return write_if_changed(path, text, updated)


def main() -> None:
    torch_dir = get_torch_dir()
    cuda_version, hip_version = get_cuda_and_hip_versions(torch_dir)

    if hip_version is not None:
        print("skipping PyTorch PR 190402 monkeypatch for ROCm torch")
        return
    if cuda_version is None:
        print("skipping PyTorch PR 190402 monkeypatch for non-CUDA torch")
        return

    print(f"reverting PyTorch PR 190402 in installed torch package: {torch_dir}")

    changed = False
    changed |= patch_invert_expr_analysis(
        torch_dir / "_inductor" / "invert_expr_analysis.py"
    )
    changed |= patch_scheduler(torch_dir / "_inductor" / "scheduler.py")

    if changed:
        print("PyTorch PR 190402 monkeypatch applied")
    else:
        print("PyTorch PR 190402 monkeypatch was already applied")


if __name__ == "__main__":
    main()
