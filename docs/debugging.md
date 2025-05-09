## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/debug_model.toml" ./run_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder memory_snapshot
```
* `--profiling.enable_memory_snapshot`: to enable memory profiling
* `--profiling.save_memory_snapshot_folder`: configures the folder which memory snapshots are dumped into (`./outputs/memory_snapshot/` by default)
	+ In case of OOMs, the snapshots will be in `./outputs/memory_snapshot/iteration_x_exit`.
	+ Regular snapshots (taken every `profiling.profile_freq` iterations) will be in `memory_snapshot/iteration_x`.

You can find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).

## Overriding Boolean Flags from `.toml` via CLI

Boolean flags are treated as **actions**. To disable a flag from the command line, use the `--no` prefix.

For example, given the following in your `.toml` file:

```toml
[profiling]
enable_memory_snapshot = true

```
You can override it at runtime via CLI with:

```bash
--profiling.no_enable_memory_snapshot
--profiling.no-enable-memory-snapshot  # Equivalent
```

> Note: `--enable_memory_snapshot=False` will **not** work. Use `--no_enable_memory_snapshot` instead.

## Debugging Config Values

To inspect how configuration values are interpreted—including those from `.toml` files and CLI overrides—run the config manager directly:

```bash
python -m torchtitan.config_manager [your cli args...]
```

For example,

```bash
python -m torchtitan.config_manager --job.config_file ./torchtitan/models/llama3/train_configs/llama3_8b.toml --profiling.enable_memory_snapshot
```

To list all available CLI flags and usage:

```bash
python -m torchtitan.config_manager --help
```

This will print a structured configuration to `stdout`, allowing you to verify that overrides are being applied correctly.

## Troubleshooting jobs that timeout

If you encounter jobs that timeout, you'll need to debug them to identify the root cause. To help with this process, we've enabled Flight Recorder, a tool that continuously collects diagnostic information about your jobs.
When a job times out, Flight Recorder automatically generates dump files on every rank containing valuable debugging data. You can find these dump files in the `job.dump_folder` directory.
To learn how to analyze and diagnose issues using these logs, follow our step-by-step tutorial [link](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).
