# Experiment Log

Cumulative log of all experiments. Never overwrite previous entries.

## Baseline — keep (4463e48)

- **Idea**: Establish baseline performance for Llama3 8B with aot_fx_trace, FSDP(4)+TP(2) on 8 GPUs.
- **Changes**: No changes. Default graph passes: tlparse logging, remove_detach, remove_identity_view, remove_identity_slice, annotate_flex_attention_for_regional_inductor, regional_inductor, cudagraph.
- **Result**: TPS=6938, MFU=40.63%, Memory=46.90GiB
- **Analysis**: This is the reference point. The graph already has regional_inductor compiling flex attention regions, and cudagraph wrapping the entire graph.
- **Lessons**: Starting point for all future experiments. Need to profile to understand where time is spent.

