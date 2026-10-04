# Selected Transform Relation Contributions

## Summary

TorchTitan centralizes composition policy for eagerly importable model config
transforms in `torchtitan/config/transform/relations.py`. Optional TorchTitan
transforms and third-party transforms cannot always appear in that module
because referring to their classes may import an unavailable dependency.

Add one class-level extension point, `contribute_relations`, that lets each
selected optional or third-party transform add its policy to the effective
relation graph. `apply_transforms` gathers these contributions automatically
before validating or executing any transform. Recipes continue to call
`apply_transforms(config, transforms)` without constructing relations for
TorchTitan-provided optional transforms such as DistMoE.

## Goals

- Preserve the optional dependency boundary: importing the normal transform
  package must not import DistMoE.
- Keep policy for eagerly importable TorchTitan transforms centralized.
- Make policy owned by an optional or third-party transform automatic whenever
  that transform is selected.
- Validate the complete effective graph before any config mutation.
- Keep caller-provided relations additive and avoid global mutable state.

## Non-goals

- Preserve the previous `run_after` and `conflicts_with` class attributes.
- Support removal or overriding of built-in relations.
- Support instance-configuration-dependent ordering or conflicts.
- Discover relations from transform modules that were not selected.

## Ownership

Relation ownership follows import boundaries:

- Relations among eagerly importable TorchTitan transforms live in
  `relations.py`.
- Relations owned by an optional TorchTitan transform live in that transform's
  module and are contributed by the transform class.
- Relations owned by a third-party transform live with that transform.
- Recipe-specific policy is supplied through the existing explicit
  `relations=` argument.

This keeps central resolution and core policy without requiring the central
module to import every optional integration.

## API

`ModelConfigTransform` gains one no-op class method:

```python
class ModelConfigTransform(ABC):
    @classmethod
    def contribute_relations(cls, relations: "TransformRelations") -> None:
        """Add composition policy owned by this transform."""
```

`TransformRelations` is imported under `TYPE_CHECKING` in `base.py` to avoid a
runtime dependency cycle.

An optional transform overrides the hook in its own module:

```python
class DistMoeTransform(ModelConfigTransform):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_conflict(cls, LoRATransform)
        relations.add_conflict(cls, TokenDispatcherTransform)
```

The hook is class-level because relation matching is type-based. Contributions
must be deterministic, idempotent, and additive. Configuration-dependent
compatibility remains normal config validation rather than relation policy.

## Resolution Flow

`apply_transforms` and `transform_model_config_` use one effective graph for an
invocation:

1. Create the built-in graph, or copy the caller-provided graph.
2. Deduplicate the selected concrete transform types while preserving their
   first-seen order.
3. Call `contribute_relations` once for each selected type.
4. Reject conflicts using the effective graph.
5. Topologically order the selected transforms using the effective graph.
6. Execute the ordered transforms.

Copying prevents automatic contributions from mutating a caller-owned graph.
`add_precedence` and `add_conflict` remain idempotent, so a contribution that
duplicates central or caller policy has no effect.

Only the subgraph induced by selected transforms must be acyclic. Relation
matching remains subclass-aware. A subclass inherits its parent's hook through
normal class-method inheritance; an override may call `super()` when it wants
to extend inherited policy.

## Error Handling

Contribution, conflict, and ordering failures happen before the first
transform executes. Exceptions raised by a contribution propagate directly,
because they indicate invalid transform policy rather than a recoverable
runtime condition.

The relation hook must not import unrelated optional features. Loading the
selected transform's own module is already part of selecting that feature, but
ordinary TorchTitan imports must remain independent of it.

## DistMoE Migration

Current DistMoE declares:

```python
DistMoeTransform.conflicts_with = (
    LoRATransform,
    TokenDispatcherTransform,
)
```

During the rebase, replace this assignment with the class method above. No
DistMoE recipe should pass a custom relation graph. Its existing conflict test
should continue to call plain `apply_transforms` and pass unchanged.

## Tests

Add focused coverage for:

- automatic contribution from a selected transform;
- contribution once per selected concrete type;
- composition with caller-provided relations without mutating the caller;
- inherited contribution behavior for subclasses;
- contributed conflicts in either transform-list order;
- contributed precedence and cycle detection;
- conflict or cycle failure before any transform executes;
- importing `torchtitan.config.transform` while `dist_moe` is blocked; and
- the existing DistMoE conflicts with LoRA and token dispatcher when the
  optional package is installed.

## Alternatives Rejected

Global decorator registration was rejected because it creates import-order
dependence and mutable process-wide state. Lazy fully qualified class names in
the central graph were rejected because they are fragile under refactoring and
require custom resolution machinery. Requiring every DistMoE recipe to call
`add_conflict` was rejected because TorchTitan-provided transforms should apply
their own invariant policy automatically.
