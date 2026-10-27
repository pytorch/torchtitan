# RFC: Consistent last-words checkpoints for TorchFT training

## Summary

We would like to discuss a last-words checkpoint path for TorchTitan's
experimental TorchFT integration. If training cannot safely continue after a
failure, an eligible surviving replica should be able to save complete committed
training state for a subsequent restart. If completeness cannot be established,
the job must explicitly refuse the save.

The paired TorchFT RFC proposes independent save coordination and Manager
snapshot hooks. This proposal covers TorchTitan's state completeness, training
transactions, checkpoint publication, replica-local cursors and process
supervision. The accompanying implementation is WIP and does not establish
production readiness or successful accelerator fault recovery.

## Motivation

Periodic checkpoints bound recoverable progress, but a surviving replica may
hold newer useful state when another replica fails. Preserving that state can
reduce lost work. A model-only dump is insufficient: optimizer moments,
scheduler progress, auxiliary state and the data positions used by each logical
replica must describe the same completed update.

Failure can arrive before an update, partway through an update, during peer
healing, or while an asynchronous checkpoint upload is in flight. A final save
needs explicit eligibility rules for each boundary, and a restart must validate
the artifact instead of trusting a directory name or a service acknowledgment.

## Example use case

1. Multiple TorchFT replica groups train with full checkpointing and logical
   replica-local dataloader state enabled.
2. A failure interrupts an attempt. The integration distinguishes an uncommitted
   attempt from an authorized or unknown optimizer transaction.
3. Where backend recovery is safe, the interrupted attempt can be discarded
   under the Manager contract. Otherwise the job requests a terminal save.
4. An eligible replica receives a writer grant and writes a full checkpoint for
   its last complete update, with bindings to every required logical cursor.
5. The supervisor independently validates the publication and confirms that
   owned processes exited before admitting a replacement generation.
6. Restart validates the chosen checkpoint and bindings, restores model and
   training state, and loads the cursor for the replica's own logical slot.

## Expected behavior and requirements

### Treat the optimizer update as a transaction

A native commit decision must precede persistent optimizer pre-hooks, optimizer
updates, scheduler advancement and auxiliary updates such as EMA. A rejected
decision must not advance these effects or completed-step counters. A timeout
with unknown commit outcome is not permission to replay an update.

The integration marks the application as updating before these effects and
publishes completion only after the full update and required synchronization.
An interrupted authorized update, partial healing, or incomplete cursor must not
produce an apparently valid checkpoint.

### Publish complete, attributable state

The proposed checkpoint adapter retains model parameters and buffers, optimizer
state, scheduler, EMA when enabled, training counters, TorchFT progress when
applicable, and logical-replica cursor bindings. The Main adapter also defines
rank-local Python, NumPy, CPU and device RNG state with an explicit schema.

Publication writes a temporary candidate, persists its contents, validates the
payload and bindings, then publishes the manifest atomically. Recovery checks
storage ranges, committed progress, configuration identity and cursor/RNG
layout before loading. A matching signature alone does not prove correct bytes
or a compatible software environment.

The older 0.3.0 candidate and the Main adapter have distinct state layouts.
Existing checkpoints without the Main RNG schema are not silently promoted to
exact-replay evidence. Functional recovery with deterministic algorithms disabled
does not establish bitwise trajectory equivalence.

### Preserve logical data ownership

Shared model progress and replica-local data progress are different state.
Restoration must select the cursor bound to the checkpoint and logical replica,
not whichever cursor directory happens to be newest. Retention must preserve
referenced cursor branches as well as model shards, and reject an incomplete
layout before DCP mutates live state.

### Define process ownership and terminal outcomes

The supervisor owns its children, save rounds and admitted generations. It
checks process identity when fencing, confirms exit before replacing a writer,
validates checkpoint storage, and distinguishes success, failure and rejected
admission. It must not terminate unrelated processes.

The implementation refreshes the clock after a blocking storage validation so
that validation time is not mistakenly charged to the worker-exit grace period.
Hard job deadlines still apply. Failed artifacts are retained for diagnosis;
retention is an explicit plan that protects dependent cursor state.

## Scope and proposed integration

| Layer | Responsibility |
| --- | --- |
| TorchFT | Existing training quorum/commit/healing, save-session service and clients |
| TorchTitan FT Manager/optimizer/TrainingEngine adapters | Configuration, transaction boundary, completion accounting and lifecycle wiring |
| TorchTitan checkpoint/control modules | Eligibility, complete DCP state, cursor/RNG binding, publication and restore checks |
| TorchTitan supervisor | Owned-process lifecycle, fencing, restart selection and retention |
| Accelerator plugin | Device synchronization, cancellation, communicator recovery and backend-specific behavior |

The candidate offers `elastic`, `last_words` and `elastic_last_words` supervisor
modes, and explicit external-process admission. Its current service deployment
is single-host loopback HTTP with shared POSIX storage and Linux supervision.
There is no guarantee of saving after process death or device-state loss, no
multi-host orchestration guarantee, and no hard recovery-latency promise.

The change is opt-in. Ordinary training must retain its existing path when
disabled. Initial support requires synchronous TorchFT, full checkpointing and
replica-local cursor support; semi-synchronous algorithms and optimizer CUDA
graphs are outside the proposed initial combination. Backend-specific kernels,
vendor toolkit patches and validation schedulers do not belong in this PR.

The WIP is rebased onto the current TrainingEngine APIs. Prior 0.3.0 normal
training and CPU checkpoint results are historical evidence, not acceptance of
this port. The latest accelerator recovery attempt failed before a committed
step; its SIGKILL source remains unresolved. The final recovery matrix,
same-input numerical comparison and paired performance measurements remain
incomplete. See [implementation and validation notes](last_words.md).

## Questions for discussion

1. Is this complete-state emergency publication contract appropriate for the
   experimental TorchFT integration?
2. Which training-boundary hooks should remain in the FT adapter, and which
   would be useful as general checkpoint lifecycle interfaces?
3. What minimum state inventory and cursor/RNG compatibility policy should a
   recoverable last-words checkpoint require?
4. How should the API distinguish recoverable pre-commit interruption from an
   unknown or partially applied optimizer transaction?
5. Should the supervisor and retention tools remain in this repository, or
   should some orchestration responsibilities belong to another package?
6. Which end-to-end recovery, numerical, disabled-path regression and shutdown
   evidence should be required before this WIP can be considered for merging?
