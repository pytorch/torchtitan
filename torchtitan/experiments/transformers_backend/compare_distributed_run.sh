#!/usr/bin/bash

# python compare_distributed_run.py --steps 5 --model-filter llama --flavor debugmodel

# python compare_distributed_run.py --steps 5 --model-filter llama --flavor debugmodel --nd_parallel 2d
debugpy-run compare_distributed_run.py --steps 5 --model-filter llama --flavor debugmodel --nd_parallel 2d
