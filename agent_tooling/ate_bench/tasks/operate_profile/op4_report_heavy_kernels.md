# OP4: Report Heavy Kernels

Profile a **7-step** training run with **Nsight Systems** and identify the **top 3
most expensive CUDA kernels by total GPU time, aggregated across all 8 ranks**.

Resume training from the released HuggingFace checkpoint. Profile **only step 7**
— steps 1–6 are warmup for cudagraph capture, NCCL handshake, and allocator
priming, so they are not representative.

**Output:** a single CSV at
```
{{HEAVY_KERNELS_DIR}}/top-kernels.csv
```
with header exactly:
```
kernel_name,total_time_ms,instances,mean_time_ms
```
and **exactly three rows**, sorted by `total_time_ms` **descending**. Also save
the raw `profile.nsys-rep` in the same directory so the result is reproducible
(the kernel names are validated by a human against the Nsight GUI's CUDA GPU
Kernel Summary).
