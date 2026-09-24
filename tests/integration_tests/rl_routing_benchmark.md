# Inter-generator routing benchmark

The configurations in `rl_routing_benchmark_config.py` run Qwen3-0.6B on
four GPUs: a TP=2 trainer and two TP=1 generator replicas. Alphabet sort
provides 1-4 turns per session and up to eight new names per turn. The only
difference between `least_loaded` and `sticky` is the inter-generator routing
strategy; `sticky` uses the router default and places new sessions via its
least-loaded fallback.

Run each configuration for at least 10 optimizer steps, with identical model
assets, dataset seed, sampling settings, and GPU topology. Use separate output
directories so checkpoints and metrics do not mix:

```bash
OTEL_SDK_DISABLED=false OTEL_METRICS_EXPORTER=jsonl VLLM_LOG_STATS_INTERVAL=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m torchtitan.rl.train \
  --module tests.integration_tests.rl_routing_benchmark_config \
  --config least_loaded \
  --hf-assets-path /path/to/Qwen3-0.6B \
  --dump-folder outputs/rl/route_least_seed42 \
  --async-loop.num-training-steps 10 \
  --rollouter.train-dataset.seed 42
```

Repeat with `--config sticky` and a separate `--dump-folder`; repeat the pair
with another dataset seed. The vLLM engine metrics are written to
`<dump-folder>/vllm_metrics/generator_{0,1}.rank0.jsonl`.
The measurement host also needed its virtualenv `bin` directory on `PATH` and
CUDA 13 `libcublas.so.13` in `LD_PRELOAD` for the local vLLM build; these are
environment-specific startup fixes applied identically to both strategies.

Calculate the overall prefix-cache hit rate as the sum of the *final cumulative*
`vllm.prefix_cache_hits` counters divided by the sum of the final cumulative
`vllm.prefix_cache_queries` counters across both generators. Do not average
per-generator percentages. For load balance, compare each generator's final
`vllm.finished_requests` and `vllm.generation_tokens`, its fraction of active
one-second snapshots with `vllm.num_running_requests > 0`, and how often
`vllm.num_waiting_requests > 0`. The queue-time histogram reports the per-request
mean as `sum / count`. Exclude engine initialization and shutdown when measuring
the fraction of busy snapshots.

Observed on two 10-step runs per strategy with two H100 generator GPUs:

| Dataset seed | Least-loaded hit rate | Sticky hit rate | Change |
| --- | ---: | ---: | ---: |
| 42 | 68.37% | 69.83% | +1.46 pp |
| 43 | 66.51% | 69.70% | +3.19 pp |

Across the active generation window, each engine had running requests in
approximately 78-84% of one-second snapshots. Request splits stayed near even:
1096/1093 vs. 1088/1070 with seed 42, and 1063/1039 vs. 1037/1033 with seed
43 (least-loaded vs. sticky). Mean queue time per engine ranged from 3.2 to
4.1 ms; waiting requests appeared in at most one sampled second per run. The
generators were active, but not persistently queue-saturated.

A shorter 1-3-turn control with at most five names per turn gave 67.11% for
least-loaded and 66.64% for sticky (one 10-step run each). Thus the benefit is
workload-dependent. The counters include shared-prompt hits as well as
continuation hits, and the default generator resets its prefix cache after
each weight update. These are stochastic end-to-end runs, not identical replay
of generated responses; the small number of seeds does not establish a
production-wide throughput or cache-hit improvement.
