---
name: auto-fuser
description: Optimize GraphTrainer regions through kernel fusion with parallel implementation agents, preserving bitwise and memory parity while targeting the local GPU roofline.
---

# Auto Fuser

You're responsible for reaching the roofline local kernel performance while preserving bitiwse parithy and memory parity with the original model through kernel fusions.

You can assume gemms, attention and communication kernels have been well optimized. So they form a partitioning boundry that partitions the computation into candidate fusible regions.

For each candidate fusible region, you propose fusion plans for this region.For each fusion plan, offload the kernel implementation to parallel sub-agents. The same criteria applies to all kernels: 1. bitwise parity 2. memory parity 3. reach roofline performance (best-effort, bottom line is original implementation).
