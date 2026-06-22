# DeepSeek V3 FlexShard Placement Policy Plan

## Goal

Make the experimental DeepSeek V3 FlexShard entry point more general without
turning it into a collection of one-off placement branches.

The production direction is:

- use `Shard` as the regular FSDP-like baseline placement;
- use `GroupedOwned` for routed expert weights;
- do not expose plain `Owned` as a supported DeepSeek V3 placement policy.

This keeps the path general enough to compare `Shard` and `GroupedOwned`, while
keeping the supported production surface focused on the placement that preserves
packed routed expert layout.

## Current Implementation

The DeepSeek V3 FlexShard path now keeps training orchestration and placement
selection separate. `deepseek_v3/parallelize.py` applies EP/AC, resolves meshes,
builds the mixed-precision policy, and calls `flex_shard()`. It does not import
concrete placement classes. `deepseek_v3/placement_policy.py` owns the FlexShard
bucket construction and concrete placements.

The default policy preserves the original behavior:

- embeddings use `Shard(0)` on the DP mesh;
- dense per-layer params use `Shard(0)` on the DP mesh:
  - attention;
  - attention norm;
  - FFN norm;
  - dense feed-forward;
  - router;
  - shared experts;
- routed expert weights use `GroupedOwned` on the expert/FSDP mesh:
  - `moe.experts.w1`;
  - `moe.experts.w2`;
  - `moe.experts.w3`;
- final norm and LM head use `Shard(0)`.

The intent is explicit in `DeepSeekV3FlexShardPolicy`, and bucket construction is
split by model region in `placement_policy.py`.

## Owned Placement Goal

Make `GroupedOwned` the only public owned-style DeepSeek V3 placement policy.
Plain `Owned` should stay private to FlexShard implementation details and
experiments.

To keep that boundary clear:

- do not add public DeepSeek V3 policy modes such as `owned_layer`,
  `owned_matrix`, or per-parameter `Owned` placement selection;
- do not advertise plain `Owned` as a supported production mode;
- do not add DeepSeek V3 tests for plain `Owned`;
- do not require users to reason about plain `Owned` when selecting the
  production routed expert path;
- do not run or verify a plain `Owned` DeepSeek V3 configuration.

## Policy Shape

The path uses a small policy object that describes placement choices by model
region:

```python
@dataclass(frozen=True)
class DeepSeekV3FlexShardPolicy:
    common: Literal["shard"] = "shard"
    routed_experts: Literal["shard", "grouped_owned"] = "grouped_owned"
    output: Literal["shard"] = "shard"
```

The initial default preserves current behavior:

```text
common         = shard
routed_experts = grouped_owned
output         = shard
```

The only supported routed expert alternatives should be:

- `shard`: FSDP-like FlexShard baseline;
- `grouped_owned`: production routed expert placement.

## Bucket Construction

`_apply_flex_shard()` delegates bucket construction to
`DeepSeekV3FlexShardPolicy.build_buckets()`. The concrete helpers live in
`deepseek_v3/placement_policy.py` and describe model regions instead of
hard-coding placement decisions in the parallelization entry point.

Implemented helpers:

```python
def _embedding_bucket(...)
def _layer_common_bucket(...)
def _layer_routed_expert_bucket(...)
def _output_buckets(...)
def _build_flex_shard_buckets(...)
```

Each helper returns one or more `BucketSpec`s. The helper names make the model
region obvious, while the placement function is chosen from the policy.

## Shard Baseline

`Shard` is supported for every region that currently participates in the
DeepSeek V3 FlexShard path.

For common params and output params, this is the only supported production mode:

```python
placement_fn=_placement_fn(0)
```

For routed experts, `routed_experts="shard"` produces an FSDP-like baseline:

```text
moe.experts.w1/w2/w3 -> Shard(0)
```

This is useful for correctness, profiler comparisons, and debugging the
FlexShard runtime without the `GroupedOwned` layout.

## GroupedOwned Production Path

`GroupedOwned` remains the production placement for packed routed expert weights.

When `routed_experts="grouped_owned"`, the routed expert bucket uses:

```python
placement_fn=_grouped_owned_expert_placement_fn
```

This path preserves the packed `w1/w2/w3` grouped expert tensors while assigning
contiguous expert ranges to owners. That is the reason to prefer `GroupedOwned`
over plain `Owned`: it can split one packed expert tensor into multiple
owner-local segments without rewriting the model into per-expert parameters.

## Private Owned Placement

Plain `Owned` should be treated as private to the FlexShard implementation and
examples.

Follow-up cleanup can make this explicit by:

- removing plain `Owned` from DeepSeek V3 policy names;
- avoiding DeepSeek V3 tests for plain `Owned`;
- documenting `GroupedOwned` as the supported owned-style routed expert mode;
- optionally renaming or hiding plain `Owned` exports if the broader FlexShard
  API is ready for that change.

The important point is that production DeepSeek V3 should not expose `Owned` as
a first-class user choice.

## Tests

Focused tests cover the supported policy matrix:

```text
routed_experts="grouped_owned"
  routed expert bucket uses GroupedOwned

routed_experts="shard"
  routed expert bucket uses Shard(0)

common="shard"
  attention/router/shared experts stay Shard(0)

output="shard"
  final norm and lm_head stay Shard(0)
```

Do not add plain `Owned` DeepSeek V3 policy tests.

## Completion Status

Completed:

1. Added `DeepSeekV3FlexShardPolicy` with defaults matching previous behavior.
2. Refactored bucket construction into model-region helpers.
3. Implemented `routed_experts="shard"` as the baseline alternative.
4. Kept `routed_experts="grouped_owned"` as the default production path.
5. Added tests for `Shard` and `GroupedOwned` only.
6. Left plain `Owned` out of the public DeepSeek V3 policy surface.
7. Moved concrete placement logic into `deepseek_v3/placement_policy.py` so
   `deepseek_v3/parallelize.py` stays placement-agnostic.

Validation:

```bash
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_deepseek_v3_config.py
```

Result:

```text
7 passed, 14 warnings, 4 subtests passed
```
