# Output schemas

## regions.json

```json
{
  "substrate": "graph",
  "setup": {"git_commit": "", "command": "", "config_deltas": {}, "gpu": "",
            "gpu_busy_during_timing": false, "hbm_gbps": 0, "bf16_gemm_tflops": 0},
  "step_budget": {"step_gpu_busy_ms": 0, "step_gpu_idle_ms": 0, "attributed_pct": 0,
                  "by_category_ms": {"gemm": 0, "attention": 0, "fake_comm": 0,
                                     "memory_bound_glue": 0, "optimizer": 0, "other": 0}},
  "regions": [{
    "id": "G-001",
    "title": "",
    "phases": ["forward", "recompute", "backward"],
    "scope": ["intra-module | cross-module | cross-layer", "recompute-backward"],
    "fusion_kind": "",
    "location": {"module_fqn_pattern": "", "graph_nodes_example": [], "source": "file:line"},
    "ops": ["aten...."],
    "kernels_per_instance": 0,
    "instances_per_step": 0,
    "shapes_dtypes": "",
    "external_bytes_per_instance": 0,
    "measured_us_per_instance": 0, "measured_ms_total": 0,
    "ideal_us_per_instance": 0, "ideal_ms_total": 0,
    "gap_ms_total": 0, "gap_pct_of_step": 0,
    "bitwise_tier": "T0 | T1 | T2 | T3", "bitwise_notes": "",
    "blockers": "",
    "evidence": "measured | estimated",
    "how_found": "",
    "overlaps_with": [],
    "notes": ""
  }],
  "no_opportunity": [{"what": "", "time_ms": 0, "reason": ""}]
}
```

Field notes:
- `regions` are ranked by `gap_ms_total`. Ids are assigned in that order, so they
  change between captures; the generated passes are tied to one capture.
- `config_deltas` is `setup.json` from `enable_fusion_finder`: pass pipeline, memory
  policy, parallelism degrees, tokens per microbatch, steps.
- `scope` has exactly one of `intra-module`, `cross-module` or `cross-layer`, plus
  `recompute-backward` when the region spans both phases.
- `measured_*` sums the kernel time of the members over all instances in the profiled
  step. `ideal_*` uses external bytes only (inputs read once, outputs written once;
  stride-0 broadcasts counted once; views counted by the bytes their consumers read),
  or flops / GEMM peak if larger.
- `blockers` starts with `NOT FUSABLE AS FOUND: ...` when a path leaves the region and
  re-enters it (see SKILL.md, section 3).
- `evidence` is `estimated` when some external size stayed symbolic; those bytes count
  as 0.
- `overlaps_with` lists regions that share a kernel node. Overlap is deliberate: one
  region per opportunity, at several granularities.
- `no_opportunity` groups the memory-bound nodes no region covers by
  `(op, phase, module pattern)`.

## fusion_regions/

One folder per region, named `g<number>_<family>` (e.g. `g003_rope`), where the family
is the region's innermost module plus its GEMM kind or single op. `__init__.py` imports
every lift pass and defines `PASSES` (region id -> (folder, pass name)) and
`DEFAULT_SELECTION`: pass names in gap order that each still find a safe site after the
ones before them.

### region.json

The region's `regions.json` entry plus:

```json
"lift": {"pass_name": "lift_fusion_region_g003", "num_sites": 12,
         "in_default_selection": true, "not_liftable_reason": ""}
```

`pass_name` is null and `not_liftable_reason` says why when the region has no lift pass.

### lift_pass.py

Liftable regions only:

```python
REGION_ID = "G-003"


def match_v0(n0):
    """One instance, anchored at its aten.unsqueeze.default node: (members, external inputs) or None."""
    if not is_op(n0, 'aten.unsqueeze.default', [N, 0]):
        return None
    n1 = n0.args[0]
    ...
    return [n7, n5, ...], [x0, x1]


# (anchor op, matcher) for each structural variant.
MATCHERS = [('aten.unsqueeze.default', match_v0)]


@register_graph_pass
def lift_fusion_region_g003(gm, example_inputs=None):
    return lift_matches(gm, REGION_ID, MATCHERS)
```

- A matcher returns the instance's member nodes (views and allocations included; T3
  regions include their GEMM) and its external inputs, or `None`.
- `is_op(node, target, args, kwargs)`: `N` marks a node slot, `S` a size (any int),
  `V` a scalar operand (any number), `D('cuda')` a device of that type (any index);
  other entries are literals compared exactly (dtypes and layouts as strings, e.g.
  `"torch.float32"`). SKILL.md section 5 says which arguments get which slot.
  `user_of(node, path, target, args, kwargs, skip)` finds a user that has `node` at
  argument `path`.
- `lift_matches` (`scripts/lift_regions.py`) checks every match with `lift_hazard`
  before outlining it.
