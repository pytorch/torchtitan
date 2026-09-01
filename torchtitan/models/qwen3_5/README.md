# Qwen3.5

TorchTitan keeps Qwen3.5 as a separate model package so its released model
flavors, Hugging Face asset paths, and CLI entry points remain stable.

Qwen3.5 and Qwen3.8 use the same underlying architecture. The shared model,
parallelization, sharding, vision, and checkpoint implementation lives in
`torchtitan/models/qwen3_8/`. This package contains only the Qwen3.5-specific
model registry and training recipes. The dependency intentionally points from
Qwen3.5 to Qwen3.8 so Qwen3.5 can be removed later without relocating shared
code.

Available model flavors:

| Flavor | Type | Role |
|---|---|---|
| `debugmodel` | Dense | Unit and integration tests |
| `debugmodel_moe` | MoE | Unit and integration tests |
| `0.8B` | Dense | Smallest released dense model |
| `2B` | Dense | Small dense model |
| `4B` | Dense | Small dense model |
| `9B` | Dense | Mid-size dense model |
| `27B` | Dense | Large dense model |
| `35B-A3B` | MoE | Smallest released MoE by active parameters |
| `122B-A10B` | MoE | Mid-size MoE |
| `397B-A17B` | MoE | Large MoE |

Example:

```bash
MODULE=qwen3_5 CONFIG=qwen35_0_8b ./run_train.sh
```
