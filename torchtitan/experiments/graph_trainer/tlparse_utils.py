# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from torch._logging import trace_structured

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        SubclassLayout,
        TracedResult,
    )


def _tlparse_log_string_artifact(name: str, payload: str) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": "string",
        },
        payload_fn=lambda: payload,
        expect_trace_id=False,
    )


def _summarize_subclass_layouts(
    layouts: dict[int, SubclassLayout],
) -> dict[int, dict[str, Any]]:
    summary = {}
    for idx, layout in layouts.items():
        meta = layout.meta
        summary[idx] = {
            "num_tensors": layout.num_tensors,
            "tensor_subclass": None if meta is None else meta.cls.__name__,
            "attrs": None if meta is None else meta.attrs,
            "outer_size": None if meta is None else list(meta.outer_size),
            "outer_stride": None if meta is None else list(meta.outer_stride),
        }
    return summary


def tlparse_log_traced_result(traced_result: TracedResult) -> None:
    """Log the traced graph and metadata summary to tlparse.

    Call this after ``trace_train_step`` / ``minimal_fx_tracer`` to emit
    both the raw FX graph and a JSON summary of the ``TracedResult``.
    """
    _tlparse_log_string_artifact(
        "make_fx_graph_traced",
        traced_result.gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        ),
    )
    summary = {
        "torch_version": __import__("torch").__version__,
        "torch_cuda_version": __import__("torch").version.cuda,
        "torch_git_version": __import__("torch").version.git_version,
        "num_flat_inputs": traced_result.num_flat_inputs,
        "num_flat_outputs": traced_result.num_flat_outputs,
        "num_static_inputs": traced_result.num_static_inputs,
        "state_fqns": traced_result.state_fqns,
        "input_subclass_layouts": _summarize_subclass_layouts(
            traced_result.input_subclass_layouts
        ),
        "output_subclass_layouts": _summarize_subclass_layouts(
            traced_result.output_subclass_layouts
        ),
        "output_spec": repr(traced_result.output_spec),
    }
    _tlparse_log_string_artifact(
        "make_fx_traced_result_summary", json.dumps(summary, indent=2, sort_keys=True)
    )
