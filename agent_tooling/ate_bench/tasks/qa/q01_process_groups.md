# Q1: Process Groups / Device Mesh

Trace the sequence of function calls from the main training entry script down to
the initialization of the PyTorch distributed process groups or DeviceMesh.
Detail the exact file paths, function names, and line numbers where the world
size and ranks for the parallel groups present in this codebase (any of DP, TP,
PP, EP, CP that this repo actually supports) are assigned.
