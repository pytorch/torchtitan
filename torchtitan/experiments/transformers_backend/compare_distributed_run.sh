#!/usr/bin/bash

if [[ "$1" == "--debug" ]]; then
    shift
    debugpy-run compare_distributed_run.py --steps 10 --model-filter llama3 --flavor debugmodel --nd_parallel 1d "$@"
else
    python compare_distributed_run.py --steps 10 --model-filter llama3 --flavor debugmodel --nd_parallel 1d "$@"
fi
