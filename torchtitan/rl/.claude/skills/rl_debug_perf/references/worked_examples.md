# Worked examples

Two GB300 runs, analyzed with the triage in [SKILL.md](../SKILL.md), plus one run's error taxonomy. Both runs waited on rollouts most of each step, for different reasons. Projections use `step_time ~= G * W / L` calibrated on each run; read them as upper bounds until a run confirms them.

## A. Concurrency-bound: 35B MoE math

Qwen3.5-35B-A3B base, DAPO, 32 prompts x 16 samples, `target_offpolicy_steps=4`, 49,152-token responses, 8 single-GPU engines, steps 4-37.

```text
step 591 s = 476 s waiting for a batch (79%) + 113 s fwd/bwd + 0.1 s optimizer; weight sync blocks ~0 s
slots = (4 + 1) x 32 = 160 groups; seats = 160 x 16 = 2,560 = 8 engines x max_num_seqs 320
engines: 185 running of 320, KV 40%, queue time 5.6 ms, ITL 41.6 ms

seats:   1,477 (58%)  running on an engine
           736 (29%)  sample done, waiting for its slowest sibling
           347 (14%)  group finished, waiting for the trainer and the weight pull

Little:  G 57.6 groups (32 trained + 25.6 zero-std dropped) x W 1,381 s / L 135 groups ~= 590 s
```

Reading it:
1. The trainer waits 79% of the step, so rollouts are the bottleneck.
2. The engines have idle seats and KV, and requests never queue.
3. At a fixed KV band, ITL is flat in running sequences (120 running: 42.7 ms, 220 running: 41.7 ms). So per-engine tok/s grows linearly with running sequences, and more sequences per engine are nearly free.
4. So the 160 slots cap the sequences: 58% of seats run while 29% wait for a slow sibling. `rollout_buffer/available_active_slots` read 0 at every step but one, which is consistent but not proof on its own.
5. Mixed groups whose recorded extremes include a truncated sample waited 2,080 s, about 49,152 tokens x 42 ms. The response cap sets the critical path for nearly half the trainable groups.

Levers, projected:
- `target_offpolicy_steps` 4 -> 6: 591 -> ~450 s (-24%), mean policy age ~2.8 -> ~3.9. Staleness looked cheap: `loss/ratio_clipped_frac` was 0.18%, and the logprob gap grew ~0.1% of its baseline per step of age.
- `num_prompts_per_train_step` 32 -> 64: G and L both double (~115 groups consumed, ~277 generating), so only slower decode at higher KV moves the step: ~740 s (ITL ~52 ms at ~81% mean KV) for twice the trained groups, ~1.6x trainable groups per hour. The trainer goes from 19% to ~31% busy. `max_num_seqs` would want (4 + 1) x 64 x 16 / 8 = 640 and caps at 512; ~370 mean running per engine fits, but peak KV projects to ~108%, so expect preemptions on 8 engines.
- 12 engines instead of 8: -8% at the same off-policy steps, -32% together with 4 -> 6.
- Response cap 49,152 -> 32,768: at most -22%, but truncation scored 0 and the base model already truncated 27% of eval answers at 49K. Rejected.
- Perfect zero-std filtering: -35% ceiling, since zero-std groups used 35% of slot-seconds. Dropping prompts by another model's difficulty label projected only -5% to -9%.

## B. Env-bound: 9B terminal agent

Qwen3.5-9B base, terminal tasks in remote sandboxes, 8 prompts x 8 rollouts, `target_offpolicy_steps=4`, 150 turns, 131K context, 4 single-GPU engines, sandbox pool 256, steps 2-22.

```text
step 623 s = 577 s waiting for a batch (93%) + 46 s fwd/bwd
slots = (4 + 1) x 8 = 40 groups: 36 generating, 4 finished and waiting
engines: ~11 running of max_num_seqs 80, queue time 0.46 ms
sandboxes: 153 of 256 leased on average

one turn, 15.2 s mean:
  4.8 s  model call                                       31%
  7.0 s  sandbox exec round trips (~3.9 per turn x 1.78 s) 46%
  2.8 s  agent-chosen sleeps beyond the round trip         19%
  0.6 s  other harness work                                 4%

group tail: the k-th of 8 rollouts finishes at 5.5, 6.9, 8.3, 9.9, 11.4, 13.4, 16.2, 21.1 min
```

Reading it:
1. The trainer waits 93% of the step; engines and sandboxes are both under-used.
2. The model is 31% of a turn, so faster generation would save at most a third of rollout time.
3. Every sandbox exec costs a flat 1.78 s, and a turn makes ~3.9 of them.
4. The 40 slots bind before the pool does, and a finished rollout waits 9.6 min on average for its slowest sibling.
5. Truncated rollouts (6.3% at the turn cap, 5.2% at the context cap) used 35% of rollout wall time and scored ~0.015.
6. 48% of groups were zero-std, and the all-0 ones were the slowest (1,595 s mean against 496 s for all-1).

Levers, projected:
- One sandbox exec per turn instead of ~3.9: -23%.
- `target_offpolicy_steps` 4 -> 6: ~-26%, ~220 sandboxes leased on average (fits 256; the projected p90 of ~265 does not).
- Turn cap 150 -> 100: -12% for 2.2% of successes.
- Ending each group at its 7th finisher: ~-23% group latency but only x0.96 per trained sample, and the cancelled rollouts are mostly long failures. Rejected.
- Train only on tasks a strong model solved at least once: -38%, but it changes the training distribution, so it is a research call.
- Trainer idle 93% is a cost lever (share or shrink the trainer), not a speed lever.

## C. Error taxonomy: 9B terminal agent

The run in B, over 3,776 rollouts. Errors are classified by their final exception; the reference harness is the benchmark's original runner.

```text
errored, by final exception
  agent broke its env   49   terminal server gone 34, wiped /tmp 4, killed PID 1 or stopped the container 4,
                             kill -9 matched its own keystroke client 3, fork exhaustion 2, no bash 1,
                             deleted its workdir before the tests 1; scored 0, as the reference harness does
  sandbox infra         50   transport drops 23, lost background job 16, lost persistent shell 7, other 4;
                             each retried once on a fresh sandbox
  agent invalid input    7   a NUL byte in the keystrokes; scored 0, not retried
  agent timeout          1   the reference harness grades the partial state; this env did not
scored 0 with no error status
  dataset bug          150   rollouts on 19 tasks whose tests import `requests`, missing from the task image
  truncation           447   max turns 260, context 187
```

Also seen in this run: empty content from a stray `</think>`, and branched rollouts when two turns with the same content had different reasoning. Earlier runs of the same env also hit services killed before grading (cleanup ran before the tests), `pkill -f` matching a long-lived exec wrapper that carried the keystrokes, `exit` ending the agent's session, and a generator crash on a prompt that filled the context window.
