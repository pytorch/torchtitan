## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/debug_model.toml" ./run_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder memory_snapshot
```
* `--profiling.enable_memory_snapshot`: to enable memory profiling
* `--profiling.save_memory_snapshot_folder`: configures the folder which memory snapshots are dumped into (`./outputs/memory_snapshot/` by default)
	+ In case of OOMs, the snapshots will be in `./outputs/memory_snapshot/iteration_x_exit`.
	+ Regular snapshots (taken every `profiling.profile_freq` iterations) will be in `memory_snapshot/iteration_x`.

You cab find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).


## Troubleshooting jobs that timeout

If you encounter jobs that timeout, you'll need to debug them to identify the root cause. To help with this process, we've enabled Flight Recorder, a tool that continuously collects diagnostic information about your jobs.
When a job times out, Flight Recorder automatically generates dump files on every rank containing valuable debugging data. You can find these dump files in the `job.dump_folder` directory.
To learn how to analyze and diagnose issues using these logs, follow our step-by-step tutorial [link](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).
