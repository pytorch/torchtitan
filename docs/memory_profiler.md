## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/debug_model.toml" ./run_llama_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder output_folder
```
* `--profiling.enable_memory_snapshot`: enable memory snapshot
* `--profiling.save_memory_snapshot_folder`: dump memory snapshots in to output folder, default under your output folder to be `./outputs/memory_snapshot`.
	+ If in case of OOMs, output folder is `memory_snapshot/iteration_x_exit`.
	+ If regularly according to `profile_freq`, output folder is `memory_snapshot/iteration_x`.

Once you have dumped the memory profiler, you will find the saved pickle files in your output folder.
To visualize the snapshot file, you can utilize the `memory_viz` tool, by either dragging and dropping the snapshot into your browser or generating its HTML file, following the [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).
