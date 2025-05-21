## SimpleFSDP

This folder includes an experimental frontend implementation for [SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile](https://arxiv.org/abs/2411.00284). SimpleFSDP is a compiler-based Fully Sharded Data Parallel (FSDP) framework, which has a simple implementation for maintenance and composability, allows full computation-communication graph tracing, and brings performance enhancement via compiler backend optimizations.

### Enable SimpleFSDP Training

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --model.name llama3_simple_fsdp --training.compile
```

### Composability Support

Some of the features require the updates from PyTorch, with which we are working on providing composability support for the following features:

| Feature | Support |
| :--------: | :--------: |
|Meta Initialization| ✅ |
|Activation Checkpointing| ✅ |
|Mixed Precision Training| ✅ |
|Tensor Parallelism| ✅ |
|Context Parallelism| ✅ |
|Pipeline Parallelism| ✅ |
|Distributed Checkpointing| 🚧 |
|Float8 Training| 🚧 |


### Citation

If you find SimpleFSDP useful, please kindly consider citing the following paper:

```latex
@article{zhang2024simplefsdp,
  title={SimpleFSDP: Simpler Fully Sharded Data Parallel with torch. compile},
  author={Zhang, Ruisi and Liu, Tianyu and Feng, Will and Gu, Andrew and Purandare, Sanket and Liang, Wanchao and Massa, Francisco},
  journal={arXiv preprint arXiv:2411.00284},
  year={2024}
}
```
