# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fill TEMPLATE.md for every model and arm, and check that the two arms' prompts differ only in the
substrate block.

    python gen.py   # writes <model>_<arm>.md next to this file
"""

import difflib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/bahuang/local/step_analysis"
REPO = f"{ROOT}/repo"

PROGRAMS = {
    "dsv3": (
        "DeepSeek-V3 671B",
        "DeepSeek-V3 671B from `torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b_dist_moe_mxfp8` "
        "(MXFP8 linears, Dist-MoE routed experts with MXFP8 grouped GEMMs, the recipe's fused MLA and "
        "SwiGLU overrides), laid out as FSDP 256 x EP 64 without pipeline parallelism, with one "
        "4,096-token microbatch per step.",
    ),
    "qwen3_30b": (
        "Qwen3 30B-A3B",
        "Qwen3 30B-A3B from "
        "`torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks.qwen3_30b_a3b_dist_moe_mxfp8_8gpu` "
        "(MXFP8 attention linears, Dist-MoE routed experts with MXFP8 grouped GEMMs, FlashAttention "
        "varlen attention), laid out as FSDP 8 x EP 8, with one 32,768-token microbatch per step.",
    ),
    "llama3_8b": (
        "Llama 3 8B",
        "Llama 3 8B from `torchtitan_recipes.models.llama3.llama3_8b(seq_len=8192)` (BF16), laid out as "
        "FSDP 8, with one 8,192-token microbatch per step.",
    ),
}

SUBSTRATE = {
    "eager": """## Your artifacts

The step ran in torchtitan's eager `Trainer`. You work from the model's Python code, the eager runtime
code, and these runtime traces:
- `artifacts/traces/iteration_10/rank0_trace.json.gz`: Kineto trace of the profiled step: CPU ops, CUDA
  kernels, input shapes and Python stacks. Every module call runs inside a `record_function` named
  `## mod <module path> [<class>] ##`.
- `artifacts/traces/rank0_execution_trace.json`: execution trace of the same step
  (`torch.profiler.ExecutionTraceObserver`): every aten op, forward and backward, with input and output
  tensor and storage IDs, shapes, strides and dtypes. `seq_id` links a backward op to the forward op it came
  from, `fw_parent` / `fw_tid` mark ops on the backward thread, and `rf_id` joins each op to its Kineto
  events.
- `artifacts/capture_setup.json` and `artifacts/train.log`: the capture's settings and log.

Substrate rules: use the model source, the eager runtime code and the artifacts above. Don't trace the model
into an FX graph (`make_fx`, `torch.compile`, `torch.export`, `torch.fx`), and don't read
`torchtitan/experiments/graph_trainer/`.""",
    "graph": """## Your artifacts

The step ran as GraphTrainer's traced joint forward + backward FX graph (`memory_policy="none"`). You work
from that graph, its node metadata, its runtime trace, and the `torchtitan/experiments/graph_trainer/` code:
- `artifacts/graph_dump/final_nodes.jsonl.gz`: one JSON record per node of the graph that runs, after all
  passes: name, op, target, schema, args, kwargs, users, val (shape, stride, dtype), stack_trace,
  nn_module_stack, custom metadata, recompute tag, autograd_backward flag and seq_nr.
  `final_graph.txt.gz` is the same graph as readable code; `pre_inductor_*` is the graph before the final
  Inductor passes.
- `artifacts/traces/iteration_10/rank0_trace.json.gz`: Kineto trace of the profiled step: CPU ops, CUDA
  kernels, input shapes and Python stacks. Every graph node runs inside
  `_RecordFunctionFast("<node name>: <op> (<file:line>)")`.
- `artifacts/codegen/`: the generated code that runs the graph with those labels; `artifacts/pass_list.txt`:
  the passes applied.
- `artifacts/capture_setup.json`, `artifacts/setup.json` and `artifacts/train.log`: the capture's settings
  and log.

Substrate rules: use the graph artifacts, the node metadata and the `torchtitan/experiments/graph_trainer/`
code. Don't read model source (`torchtitan/models/`, `torchtitan/components/`, `torchtitan_recipes/`) to
find candidates. You may open a source line that a node's stack_trace points to, but only to name a region
you already found.""",
}


def main():
    template = open(os.path.join(HERE, "TEMPLATE.md")).read()
    prompts = {}
    for model, (name, program) in PROGRAMS.items():
        for arm, block in SUBSTRATE.items():
            sandbox = f"{ROOT}/{model}_{arm}"
            text = (
                template.replace("{SUBSTRATE}", block)
                .replace("{MODEL_NAME}", name)
                .replace("{PROGRAM}", program)
                .replace("{SANDBOX}", sandbox)
                .replace("{REPO}", REPO)
            )
            assert "{" + "SANDBOX}" not in text and "{MODEL_NAME}" not in text
            prompts[(model, arm)] = text
            with open(os.path.join(HERE, f"{model}_{arm}.md"), "w") as f:
                f.write(text)
    # Symmetry: after normalizing the sandbox path, eager and graph differ only inside the substrate block.
    for model in PROGRAMS:
        e = prompts[(model, "eager")].replace(f"{model}_eager", "<sandbox>")
        g = prompts[(model, "graph")].replace(f"{model}_graph", "<sandbox>")
        e_lines = e.replace(SUBSTRATE["eager"], "<SUBSTRATE>").splitlines()
        g_lines = g.replace(SUBSTRATE["graph"], "<SUBSTRATE>").splitlines()
        diff = [
            d
            for d in difflib.unified_diff(e_lines, g_lines, lineterm="", n=0)
            if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
        ]
        assert not diff, (model, diff)
        print(f"{model}: eager and graph prompts identical outside the substrate block")


if __name__ == "__main__":
    main()
