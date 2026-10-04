# Plan: Selected Transform Relation Contributions

**Goal**: Automatically merge relation policy from selected optional and third-party transforms into TorchTitan's central transform graph.
**Architecture**: Keep eager TorchTitan policy in `relations.py`; selected transform classes add optional policy through one `contribute_relations` class method before validation or execution.
**Tech Stack**: Python, unittest/pytest, TorchTitan model config transforms.

## Step 1: Rebase onto current upstream main

**Files**: Existing branch commits only.

1. Confirm the worktree is clean and record the current head.
2. Rebase the branch onto `upstream/main`, which contains DistMoE.
3. Inspect transform-related rebased changes and resolve only semantic overlap.

## Step 2: Add failing selected-contribution tests

**File**: `tests/unit_tests/cpu/test_transforms.py`

### 2a. Write failing tests

Add local transform classes whose class hook contributes precedence and conflict
relations. Test automatic conflict rejection, precedence, contribution once per
selected concrete type, subclass inheritance, and non-mutation of an explicit
caller graph.

### 2b. Verify RED

```bash
condatorch
taskset -c 0-7 pytest tests/unit_tests/cpu/test_transforms.py -x
```

The tests must fail because `ModelConfigTransform` and `apply_transforms` do not
yet consume selected-transform contributions.

## Step 3: Implement effective relation resolution

**Files**:

- `torchtitan/config/transform/base.py`
- `torchtitan/config/transform/relations.py`
- `torchtitan/config/transform/apply.py`

### 3a. Write minimal implementation

1. Add the no-op `ModelConfigTransform.contribute_relations` class method.
2. Add `TransformRelations.copy`.
3. Build an effective graph by copying explicit relations or creating the
   default graph, deduplicating selected concrete types, and invoking each hook.
4. Use the effective graph for conflict rejection and ordering.

### 3b. Verify GREEN

Run the Step 2 command and confirm all transform tests pass.

## Step 4: Migrate DistMoE composition policy

**Files**:

- `torchtitan/config/transform/dist_moe.py`
- `tests/unit_tests/cpu/test_dist_moe.py`

### 4a. Verify existing test is RED

Run the existing DistMoE incompatible-transform test after Step 3 but before
migrating its legacy `conflicts_with` assignment. It must fail because the new
resolver deliberately does not consume the removed legacy attribute.

```bash
condatorch
taskset -c 0-7 pytest tests/unit_tests/cpu/test_dist_moe.py -x -k incompatible_transforms
```

### 4b. Implement migration

Replace the legacy class attribute assignment with a
`DistMoeTransform.contribute_relations` override that adds conflicts with LoRA
and `TokenDispatcherTransform`.

### 4c. Verify GREEN

Run the Step 4 test command. If the optional package is unavailable, record the
skip and rely on the synthetic core test plus import-boundary coverage.

## Step 5: Document the extension API

**File**: `torchtitan/config/transform/README.md`

Document central relation ownership, selected-transform contributions, and a
third-party example. State that contribution is automatic and additive.

## Step 6: End-to-end verification

Run:

```bash
condatorch
taskset -c 0-7 pytest tests/unit_tests/cpu/test_transforms.py -x
taskset -c 0-7 pytest tests/unit_tests/cpu/test_dist_moe.py -x
pre-commit run --all-files
git diff --check
```

Inspect the final diff for optional-import regressions and unrelated changes.

## Dependencies

| Group | Steps | Can Parallelize |
|---|---|---|
| 1 | Step 1 | No |
| 2 | Steps 2-3 | No; RED-GREEN sequence |
| 3 | Step 4 | No; depends on Step 3 |
| 4 | Step 5 | No; documents final API |
| 5 | Step 6 | No |
