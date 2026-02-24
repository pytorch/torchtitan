We welcome the community to submit reproducible benchmarking results.

## Submission Guidelines

A submission should be a file / files including the following information

1. Entity, which could be your name, GitHub username, company, university, team, etc.
2. The model or theme of benchmarking, e.g. Llama 3.1, Async TP.
3. The hardware setup, including the types of GPUs, interconnections, etc.
4. The actual performance report with training configs, e.g. via
   - Python config files / commandline arguments
   - complete configs, which can be found in the log with [`--print_config`](https://github.com/pytorch/torchtitan/blob/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc/torchtitan/config_manager.py#L47) turned on (preferred as the default value not shown in config files or specified in commandline could change from time to time)
5. The versions and date/time of `torchtitan`, `torch`, `torchao`, or any relevant dependencies.
6. Other notes which could help reproduce the results.

The name of the file should follow the format of
```
[model/theme]_[hardware]_[date/time]_[entity].md
```
For example, `llama3.1_h100_202412_pytorch.md`, `asynctp_256xh100_20250613_alice+bob.md`.

An example can be found at [llama3_h100_202412_torchtitan.md](./llama3_h100_202412_torchtitan.md).
