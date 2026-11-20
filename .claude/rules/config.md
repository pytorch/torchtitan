---
description: Rules for the config system
globs: torchtitan/**
---

# Config System Rules

## Each Component Owns Its Config
- A component is a `Configurable` with a nested `Config`; build it with
  `X.Config.build(...)`, not by constructing it from loose kwargs.
- A function or class takes only the config it reads (its own `X.Config` plus
  explicit arguments), never the whole trainer config. Passing the root config
  around is information leakage.
- New fields go on the config of the component they configure. A field with no
  other owner goes in `torchtitan/config/configs.py` only after checking with the
  maintainers.

## Select Implementations by Config Type
- Alternative implementations are sibling `Configurable` classes selected by
  which `X.Config` is placed in the tree -- not by a `mode` string, a name lookup,
  or a boolean flag.
- An optional component is `X.Config | None`; `None` means disabled. Don't add
  `enable_*` / `enabled` booleans or `"none"` / `"disabled"` sentinels.
- Use `Literal[...]` (not `Enum`, not bare `str`) for a small fixed set of values.

## Build Complete, Never Mutate Afterwards
- Values derived from the model or parallelism (sharding, vocab size, ...) are
  set where the config node is built (e.g. `build_model_config(flavor, *,
  parallelism=...)`). Don't patch a finished config tree at runtime or in the
  trainer, and don't add `update_from_config` / `set_*_()` walkers.
- Model architecture is part of the model definition: set it in the model's
  flavors, not per run.

## Few Knobs
- Expose only options a verified recipe uses. torchtitan is not a config broker
  for every argument of the APIs it calls; don't forward free-form `**kwargs`
  dicts.
- No new CLI flags: training is configured by Python recipes plus `--override`.
- Every new knob needs an answer to "how would a user decide how to set this?".
  If a value is always the same in practice, hardcode it. If a feature should
  always be on, don't add a flag for it.
- Prefer as few environment variables as possible; each one must be justified
  and documented.

## Defaults and Validation
- Required fields have no default. Defaults must be justified, not arbitrary
  (no magic numbers without a comment).
- Validate in `__post_init__` of the owning config (config time), not in
  `parallelize`, constructors, or the training loop.
- An option that is unsupported in some code path raises `ValueError`; don't warn
  and silently ignore it.

## Safety
- When config values are passed through to external APIs (e.g. DataLoader kwargs),
  ensure all values are safe for all code paths (e.g. what happens when
  `num_workers == 0`?).
