# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Audit finder and judge transcripts for leaks and rule violations.

    python audit.py <out.json> <label>=<transcript.jsonl>[:<sandbox>:<arm>] [...]

Flags: forbidden tools (Skill, Agent, Workflow, WebFetch, WebSearch); file paths or shell commands that
touch anything outside the agent's sandbox, the repo snapshot, installed package source or /tmp; GPU use;
and substrate rules (eager: no FX tracing, no graph_trainer reads; graph: no model-source reads beyond
naming). Also counts tool calls, wall time and tokens.
"""

import json
import re
import sys

ROOT = "/home/bahuang/local/step_analysis"
ALLOWED = (
    f"{ROOT}/repo",
    "/home/bahuang/local/pytorch",
    "/home/bahuang/local/dist_moe",  # installed dist_moe package source (allowed by the prompt)
    "/home/bahuang/.conda/envs",
    "/tmp",
    "/usr/",
    "/dev/null",
)
FORBIDDEN_TOOLS = {"Skill", "Agent", "Workflow", "WebFetch", "WebSearch", "SendMessage"}
GPU_PAT = re.compile(
    r"nvidia-smi|CUDA_VISIBLE_DEVICES|torch\.cuda\.|\.cuda\(|device=['\"]cuda|torchtitan\.train"
)
PATH_PAT = re.compile(r"(/home/[^\s'\"`;|&)>]+)")


def _paths(text):
    return PATH_PAT.findall(text or "")


def audit(path, sandbox, arm):
    allowed = (*ALLOWED, sandbox) if sandbox else ALLOWED
    flags, tools, first, last = [], {}, None, None
    # Files the harness saved this agent's own oversized tool outputs to ("Full output saved to: ...").
    own_spills = set()
    for line in open(path):
        d = json.loads(line)
        m = (d.get("message") or {}).get("content") if d.get("type") == "user" else None
        if isinstance(m, list):
            for c in m:
                if c.get("type") == "tool_result":
                    own_spills.update(
                        re.findall(
                            r"saved to: (\S+\.txt)", json.dumps(c.get("content"))
                        )
                    )
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    for line in open(path):
        d = json.loads(line)
        ts = d.get("timestamp")
        if ts:
            first = first or ts
            last = ts
        msg = d.get("message") or {}
        u = msg.get("usage") if isinstance(msg, dict) else None
        if u:
            for k in usage:
                usage[k] += u.get(k, 0) or 0
        if d.get("type") != "assistant":
            continue
        for c in msg.get("content", []):
            if c.get("type") != "tool_use":
                continue
            name, inp = c["name"], c.get("input", {})
            tools[name] = tools.get(name, 0) + 1
            text = json.dumps(inp)
            if name in FORBIDDEN_TOOLS:
                flags.append(
                    {"kind": "forbidden_tool", "tool": name, "input": text[:300]}
                )
            for p in _paths(text):
                if not p.startswith(allowed) and p.rstrip(";") not in own_spills:
                    flags.append(
                        {
                            "kind": "outside_path",
                            "tool": name,
                            "path": p,
                            "input": text[:300],
                        }
                    )
            if (
                name == "Bash"
                and ROOT not in inp.get("command", "")
                and inp.get("command", "").split()[:1] != ["date"]
            ):
                flags.append(
                    {
                        "kind": "review_no_sandbox_path",
                        "tool": name,
                        "input": text[:300],
                    }
                )
            if GPU_PAT.search(text):
                flags.append({"kind": "gpu", "tool": name, "input": text[:300]})
            if arm == "eager" and re.search(
                r"(?<!-path \./)torchtitan/experiments/graph_trainer|make_fx|torch\.compile\(|torch\.export\.|torch\.fx\.",
                text,
            ):
                flags.append(
                    {"kind": "substrate_rule", "tool": name, "input": text[:300]}
                )
            if arm == "graph" and re.search(
                r"repo/torchtitan/models/|repo/torchtitan/components/|repo/torchtitan_recipes/",
                text,
            ):
                flags.append(
                    {"kind": "substrate_rule_review", "tool": name, "input": text[:300]}
                )
    return {
        "transcript": path,
        "tool_calls": tools,
        "num_tool_calls": sum(tools.values()),
        "first": first,
        "last": last,
        "usage": usage,
        "flags": flags,
    }


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    res = {}
    for s in specs:
        label, rest = s.split("=", 1)
        parts = rest.split(":")
        path = parts[0]
        sandbox = parts[1] if len(parts) > 1 else None
        arm = parts[2] if len(parts) > 2 else None
        r = audit(path, sandbox, arm)
        res[label] = r
        kinds = {}
        for f in r["flags"]:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        print(
            f"{label}: {r['num_tool_calls']} tool calls, {r['first']} -> {r['last']}, flags {kinds}"
        )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
