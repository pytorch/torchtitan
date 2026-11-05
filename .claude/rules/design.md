---
description: Structural and design rules for torchtitan code (distilled from the maintainers' refactors)
globs: torchtitan/**, torchtitan_recipes/**
---

# Design Rules

## Minimalism
- Every new symbol needs a current consumer. Delete unused fields, arguments,
  branches, modes, and files rather than keeping them "for later".
- Don't add a function for a single call site, a wrapper that only forwards, or a
  base class with a single subclass. A method that doesn't use `self` is a
  function, and it belongs near its only caller.
- Prefer one way to do each thing. If two options exist and one is strictly
  better, remove the other. If a combination of features isn't needed, ban it
  with a clear error instead of supporting it.
- Debug-only and eager fallback paths don't deserve optimization or extra
  configuration when compiled/kernel paths exist.
- Hacky integration (functions named `install_*`, `patch_*`, monkeypatching,
  threading tricks) usually signals the wrong layer. Look for the first-principles
  design, or the upstream API that should exist.

## Ownership and Boundaries
- Group code by role or owner, not by technical layer: a service's code lives in
  one package, and engine- or framework-specific code (vLLM, Monarch, ...) lives in
  a subpackage of it. No catch-all `utils.py` / `types.py`; a type lives in the
  module that creates it.
- Service cores are plain, framework-free classes; actor/engine wrappers are thin
  adapters around them. Prefer composition where users may subclass the core.
- Heavy or optional dependencies are imported only inside their subpackage (and
  lazily for default configs). Importing a package has no side effects: no heavy
  re-exports, no environment mutation.
- A component receives only what it needs. Don't pass the whole trainer,
  the whole training config, or a vendor-specific concept up the call chain.
- Model identity is the model class: no parallel "spec" objects that pair a model
  with its adapter, parallelize function, or pipeline function.

## Extensibility Without Speculation
- When there will clearly be more than one implementation (a second engine, a
  second load balancer), define the interface by its consumers and select
  implementations by config type. Ask "if tomorrow I add another X, do I have to
  edit this code again?".
- Until then, don't build registries, plugin systems, or abstract base classes.
  "Registry" is only an accurate name if something registers.
- The override mechanism (`torchtitan/config/OVERRIDE.md`) is for out-of-repo
  extension and opt-in in-repo examples. Features the repo maintains and
  guarantees are first-class (configs, converters, transforms), not overrides.

## Folder Conventions
- Every model folder has the same skeleton (`flavors.py`, `model.py`,
  `sharding.py` if needed, `state_dict_adapter.py`, `__init__.py`, `README.md`).
- Flatten folders that hold one or two files.
- Recipes live in `torchtitan_recipes/`. Docs live next to the code they describe.
- Pin new dependencies; put dev/test-only ones in `requirements-dev.txt`, and
  example-only ones at the example level, not repo-wide.
