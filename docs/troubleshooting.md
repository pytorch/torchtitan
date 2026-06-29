# Troubleshooting Guide

This guide covers common issues you may encounter when using TorchTitan and their solutions.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Training Issues](#training-issues)
- [Configuration Issues](#configuration-issues)
- [Distributed Training](#distributed-training)
- [Performance Issues](#performance-issues)
- [Getting Help](#getting-help)

---

## Setup and Installation

### GPU Not Detected

**Problem:** PyTorch doesn't detect your GPU
```
RuntimeError: No CUDA GPUs are available
```

**Solutions:**
1. Verify GPU is available:
   ```bash
   nvidia-smi
   ```

2. Check PyTorch CUDA installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

3. Reinstall PyTorch with correct CUDA version:
   ```bash
   # Check your CUDA version
   nvcc --version

   # Install matching PyTorch (example for CUDA 12.6)
   pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
   ```

### Tokenizer Download Fails

**Problem:** Cannot download tokenizer from HuggingFace
```
HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**
1. Verify HuggingFace access:
   - Visit https://huggingface.co/meta-llama/Llama-3.1-8B
   - Accept the license agreement
   - Generate a token at https://huggingface.co/settings/tokens

2. Download tokenizer with token:
   ```bash
   python scripts/download_hf_assets.py \
     --repo_id meta-llama/Llama-3.1-8B \
     --assets tokenizer \
     --hf_token=YOUR_TOKEN_HERE
   ```

### Import Errors

**Problem:** Module not found errors
```
ModuleNotFoundError: No module named 'torchtitan'
```

**Solutions:**
1. Install in development mode:
   ```bash
   pip install -e .
   ```

2. Or add to PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Pre-commit Hooks Fail

**Problem:** Pre-commit checks fail on commit

**Solutions:**
1. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. Fix issues automatically:
   ```bash
   pre-commit run --all-files
   ```

3. Update hooks if outdated:
   ```bash
   pre-commit autoupdate
   ```

---

## Training Issues

### Out of Memory (OOM) Errors

**Problem:** CUDA out of memory during training
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in your config:
   ```toml
   [training]
   batch_size = 1  # Try smaller batch size
   ```

2. **Enable activation checkpointing**:
   ```toml
   [activation_checkpoint]
   mode = 'selective'  # or 'full'
   ```

3. **Use gradient accumulation** instead of larger batches:
   ```toml
   [training]
   batch_size = 1
   global_batch_size = 8  # Effective batch size through accumulation
   ```

4. **Enable FSDP2 with CPU offload**:
   ```toml
   [model]
   fsdp_cpu_offload = true
   ```

5. **Profile memory** to identify bottleneck:
   ```bash
   CONFIG_FILE="./train_configs/debug_model.toml" ./run_train.sh \
     --profiling.enable_memory_snapshot \
     --profiling.save_memory_snapshot_folder memory_snapshot
   ```
   Then visualize at https://pytorch.org/memory_viz

### Training Hangs at Start

**Problem:** Training process hangs without error messages

**Solutions:**

1. **Enable debug logging**:
   ```bash
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   export NCCL_DEBUG=INFO
   ```

2. **Check GPU availability** on all nodes:
   ```bash
   nvidia-smi
   ```

3. **Verify network connectivity** (multi-node):
   ```bash
   # Test from node 0 to node 1
   ping <node1-ip>
   ```

4. **Start with debug model**:
   ```bash
   CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh
   ```

### Checkpoint Loading Fails

**Problem:** Cannot load checkpoint
```
FileNotFoundError: Checkpoint not found
```

**Solutions:**

1. **Verify checkpoint path** in config:
   ```toml
   [checkpoint]
   folder = "./outputs/checkpoint"
   ```

2. **Check checkpoint compatibility**:
   - Ensure same model architecture
   - Verify parallelism settings match

3. **Use checkpoint conversion** if from HuggingFace:
   ```bash
   python scripts/checkpoint_conversion/convert_from_hf.py \
     --model_name llama3 \
     --hf_checkpoint_path /path/to/hf \
     --output_path ./checkpoint
   ```

### Loss is NaN or Diverging

**Problem:** Loss becomes NaN or increases instead of decreasing

**Solutions:**

1. **Reduce learning rate**:
   ```toml
   [optimizer]
   lr = 1e-5  # Try smaller learning rate
   ```

2. **Enable gradient clipping**:
   ```toml
   [training]
   max_norm = 1.0  # Clip gradients
   ```

3. **Check data quality**:
   - Verify tokenizer is working correctly
   - Check for corrupted data samples

4. **Use deterministic mode** for debugging:
   ```toml
   [training]
   deterministic = true
   seed = 42
   ```

---

## Configuration Issues

### Boolean Flag Overrides Not Working

**Problem:** Cannot disable a boolean flag from CLI

**Solution:** Use `--no` prefix for boolean flags:

```bash
# WRONG - This won't work
--profiling.enable_memory_snapshot=false

# CORRECT - Use --no prefix
--profiling.no_enable_memory_snapshot
# or
--profiling.no-enable-memory-snapshot
```

See [debugging.md](debugging.md#overriding-boolean-flags-from-toml-via-cli) for more details.

### Config Validation Errors

**Problem:** Configuration validation fails

**Solutions:**

1. **Debug config values**:
   ```bash
   python -m torchtitan.config.manager \
     --job.config_file ./path/to/config.toml \
     [your other args...]
   ```

2. **Check for typos** in config keys

3. **Verify required fields** are present

### Conflicting Parallelism Settings

**Problem:** Parallelism dimensions don't multiply to GPU count

**Solutions:**

1. **Verify parallelism math**:
   ```
   total_gpus = dp_size * tp_size * pp_size * cp_size
   ```

2. **Example for 8 GPUs**:
   ```toml
   [parallelism]
   dp = 4
   tp = 2
   pp = 1
   cp = 1
   # 4 * 2 * 1 * 1 = 8 ✓
   ```

---

## Distributed Training

### Multi-Node Training Not Starting

**Problem:** Multi-node training fails to initialize

**Solutions:**

1. **Check network configuration**:
   ```bash
   # Set correct master address
   export MASTER_ADDR="node0-hostname"
   export MASTER_PORT=29500
   ```

2. **Verify all nodes can communicate**:
   ```bash
   # From each node, ping master
   ping $MASTER_ADDR
   ```

3. **Check firewall settings** - ensure port is open

4. **Use correct torchrun arguments**:
   ```bash
   torchrun \
     --nnodes=2 \
     --nproc_per_node=8 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     torchtitan/train.py --job.config_file config.toml
   ```

### NCCL Errors

**Problem:** NCCL communication errors

**Solutions:**

1. **Set NCCL timeout**:
   ```bash
   export NCCL_TIMEOUT=1800
   ```

2. **Use correct network interface**:
   ```bash
   export NCCL_SOCKET_IFNAME=eth0  # or your network interface
   ```

3. **Enable NCCL debugging**:
   ```bash
   export NCCL_DEBUG=INFO
   ```

---

## Performance Issues

### Low GPU Utilization

**Problem:** GPUs are underutilized during training

**Solutions:**

1. **Increase batch size** (if memory allows):
   ```toml
   [training]
   batch_size = 4  # Increase if you have memory headroom
   ```

2. **Enable torch.compile**:
   ```toml
   [model]
   compile = true
   ```

3. **Check data loading** isn't the bottleneck:
   - Monitor `nvidia-smi` during training
   - If GPU usage is spiky, data loading may be slow

4. **Profile the training**:
   ```bash
   CONFIG_FILE="./train_configs/debug_model.toml" ./run_train.sh \
     --profiling.enable_profiling
   ```

### Slow Training Speed

**Problem:** Training is slower than expected

**Solutions:**

1. **Enable all optimizations**:
   ```toml
   [model]
   compile = true

   [activation_checkpoint]
   mode = 'selective'  # Balance memory and speed
   ```

2. **Check TFLOPs and MFU** in logs:
   - Compare with benchmark numbers in `benchmarks/`

3. **Verify parallelism strategy**:
   - Too much parallelism can add communication overhead
   - Find the right balance for your hardware

---

## Getting Help

### Where to Ask Questions

1. **PyTorch Forums** - Best for general questions
   - https://discuss.pytorch.org/c/distributed/torchtitan/44

2. **GitHub Issues** - For bugs and feature requests
   - https://github.com/pytorch/torchtitan/issues

3. **GitHub Discussions** - For design discussions
   - https://github.com/pytorch/torchtitan/discussions

### How to Report Issues

When reporting an issue, include:

1. **Environment information**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   ```

2. **Minimal reproducible example**:
   - Smallest config that reproduces the issue
   - Exact command you ran

3. **Full error traceback**

4. **What you've already tried**

### Useful Debugging Commands

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# List available GPUs
nvidia-smi --list-gpus

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check config interpretation
python -m torchtitan.config.manager --job.config_file config.toml

# Run with maximum debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
CONFIG_FILE="./train_configs/debug_model.toml" ./run_train.sh
```

---

## Additional Resources

- [Debugging Documentation](debugging.md) - Memory profiling and debugging
- [FSDP Documentation](fsdp.md) - FSDP2 configuration and usage
- [Checkpoint Documentation](checkpoint.md) - Checkpointing best practices
- [Performance Benchmarks](../benchmarks/) - Expected performance numbers

---

**Note:** This guide is continuously updated. If you encounter an issue not covered here, please contribute by adding it!
