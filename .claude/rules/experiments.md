---
description: Rules for the experiments folder
globs: torchtitan/experiments/**
---

# Experiments Folder Rules

## Looser Standards, Still Linted
Code in experiments has more flexibility than core, but must still pass
`pre-commit run --all-files`. No exceptions for linting.

## Don't Modify Core for Experiments
If an experiment needs behavior that core doesn't provide, work around it in the
experiment folder. Don't change core torchtitan code to accommodate experiment needs
(e.g. don't add `if experiment_x:` branches to `train.py` or core modules).
The experiments folder is "more-or-less a hack" by design.

## Use TorchTitan's Config System
Use torchtitan's existing config and job config infrastructure. Don't introduce
custom argument parsing or parallel config systems.

## Separate Concerns
Keep distinct features in separate folders (e.g. inference code separate from RL
work). Don't bundle unrelated functionality just because it's all "experimental."
