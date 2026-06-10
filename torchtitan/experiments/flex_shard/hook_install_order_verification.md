# FlexShard hook install order verification

## Question

Can `flex_shard()` install batched all-gather hooks before applying
`reshard_after_forward` activation-checkpoint wrapping?

Current order:

```text
register parameter accessors
apply reshard-after-forward activation-checkpoint wrapping
install batched all-gather hooks
```

Proposed alternate order:

```text
register parameter accessors
install batched all-gather hooks
apply reshard-after-forward activation-checkpoint wrapping
```

## Hypothesis

The alternate order should work for well-formed per-layer buckets because
forward hooks registered on a module stay with that module after it is wrapped
by `CheckpointWrapper`. During checkpoint replay, the wrapper calls the inner
module, so the inner module's pre-forward hook should still rerun.

However, the alternate order may weaken validation for unsafe root-level
`reshard_after_forward` buckets. In the current order, FlexShard first creates
checkpoint-wrapped children, then hook resolution can detect that a root hook
would not replay when only a checkpointed child recomputes. If hooks are
installed before wrapping, that check cannot see the checkpointed children that
FlexShard is about to create.

## Verification Plan

1. Temporarily move `_install_batched_allgather_hooks(storages, module_param_map)`
   before `_apply_reshard_after_forward(module, reshard_storages)`.
2. Run a positive per-layer `reshard_after_forward=True` training test:

   ```bash
   pytest -q \
     torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_with_activation_checkpointing
   ```

3. Run the unsafe root-bucket rejection test:

   ```bash
   pytest -q \
     torchtitan/experiments/flex_shard/tests/test_flex_shard_api.py::TestFlexShardAPI::test_reshard_after_forward_requires_replayable_bucket_hook
   ```

4. Restore the original code order.

## Results

### Iteration 1: install hooks before applying reshard-after-forward wrapping

Temporary patch:

```text
register parameter accessors
install batched all-gather hooks
apply reshard-after-forward activation-checkpoint wrapping
```

Positive per-layer training test:

```bash
pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_with_activation_checkpointing
```

Result:

```text
1 passed in 8.88s
```

Unsafe root-bucket rejection test:

```bash
pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_api.py::TestFlexShardAPI::test_reshard_after_forward_requires_replayable_bucket_hook
```

Result:

```text
FAILED ... AssertionError: RuntimeError not raised
```

The alternate order works for well-formed per-layer buckets, but it breaks the
validation that rejects unsafe root-level `reshard_after_forward` buckets.

### Iteration 2: restore current order

Restored:

```text
register parameter accessors
apply reshard-after-forward activation-checkpoint wrapping
install batched all-gather hooks
```

Re-ran the unsafe root-bucket rejection test:

```text
1 passed in 2.63s
```

## Conclusion

`_install_batched_allgather_hooks()` should stay after
`_apply_reshard_after_forward()`.

The reason is not that hooks must be installed after wrapping for replay in the
well-formed per-layer case. Hooks installed before wrapping remain on the inner
module and replay correctly.

The reason is validation: after `_apply_reshard_after_forward()`, hook
resolution can see checkpoint-wrapped children. That lets
`resolve_bucket_forward_hook_module()` reject a root-level bucket whose hook
would not rerun when activation checkpointing recomputes only a child module.
If hooks are installed first, that future checkpoint structure is not visible,
so the unsafe root hook is accepted.

Recommended code comment near the current order:

```python
# Apply reshard-after-forward wrapping before installing bucket hooks so hook
# resolution sees checkpoint-wrapped children. The hook is registered on the
# inner _checkpoint_wrapped_module, which keeps it inside the replayed AC
# region, and root-level buckets that would not replay during child recompute
# can be rejected.
```
