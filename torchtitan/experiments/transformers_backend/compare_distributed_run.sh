#!/usr/bin/bash

python compare_distributed_run.py --steps 5 --model-filter llama3 --flavor debugmodel --nd_parallel 1d --verbose

# debugpy-run compare_distributed_run.py --steps 5 --model-filter llama3 --flavor debugmodel --nd_parallel 0d
