# RL weight synchronization

The controller uses `WeightSyncManager` to publish each optimized trainer
policy through TorchStore, update the generators, and admit new rollouts only
after the generator update finishes. Trainer staging and publication for policy
N overlap forward/backward for step N+1. On each generator, the network prefetch
can overlap ongoing generation before the local model update. Both overlaps
preserve the ordering described below.

```text
Trainer:    forward/backward N
                |
Controller: wait for trainer push N-1
                |
Trainer:    optimizer N produces policy N
                |
Controller: wait for generator update N-1
                |
Controller: start asynchronous weight sync N
                |
                +--> Trainer: stage policy N
                |        --> TorchStore: put_state_dict N
                |        --> Generator: update to policy N
                |
                +--> Trainer: forward/backward N+1
                         --> Controller: wait for trainer push N
                         --> Trainer: optimizer N+1 produces policy N+1
                         --> Controller: wait for generator update N
                         --> Controller: start asynchronous weight sync N+1
```

## Flow

- Weight sync N and trainer forward/backward N+1 run concurrently.
- The trainer stages policy N and calls TorchStore `put_state_dict`. The
  generator update then prefetches policy N into pinned CPU memory and applies
  it to the GPU model. Prefetch can overlap generation; application happens at
  a controlled engine boundary.
- Completing `put_state_dict N` releases `wait_prev_push`, allowing trainer
  optimizer N+1 to mutate the model.
- Completing generator update N releases `wait_prev_pull`, allowing the
  controller to reuse the staging buffers and TorchStore key for weight sync
  N+1.

## Why the overlap is safe

- The staging stream waits for optimizer step N before copying. It can overlap
  forward/backward N+1 because forward/backward does not mutate persistent model
  state.
- The CUDA-event wait and TorchStore PUT run together in an `asyncio.to_thread`
  worker. This lets them progress while synchronous forward/backward blocks the
  trainer actor's event loop.
- Before optimizer step N+1 can mutate the model, the controller awaits
  `wait_prev_push()`. The staged snapshot therefore cannot race with a weight
  update.
- Before push N+1 can reuse the staging buffers or TorchStore key, the controller
  awaits `wait_prev_pull()`. New rollouts are admitted only after the generators
  have applied the updated policy.
