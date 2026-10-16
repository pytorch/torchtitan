We welcome the community to submit reproducible benchmarking results.

## Submission Guidelines

A submission should be a file / files including the following information

1. Entity, which could be your name, GitHub username, company, university, team, etc.
2. The model or theme of benchmarking, e.g. Llama 3.1, Async TP.
3. The hardware setup, including the types of GPUs, interconnections, etc.
4. The actual performance report with training configs, e.g. via
   - Python config files / commandline arguments
   - complete resolved configs, printed by passing `--print-config` to
     `run_train.sh` (preferred because defaults can change over time)
5. The versions and date/time of `torchtitan`, `torch`, `torchao`, or any relevant dependencies.
6. Other notes which could help reproduce the results.

The name of the file should follow the format of
```
[model/theme]_[hardware]_[date/time]_[entity].md
```
For example, `llama3.1_h100_202412_pytorch.md`, `asynctp_256xh100_20250613_alice+bob.md`.

An example can be found at [llama3_h100_202412_torchtitan.md](./llama3_h100_202412_torchtitan.md).
