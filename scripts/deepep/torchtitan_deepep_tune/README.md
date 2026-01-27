# TorchTitan DeepEP Tuner - Single Node

Clean, organized DeepEP configuration tuning for TorchTitan single-node setups.

## 📁 Directory Structure

```
torchtitan_deepep_tune/
├── tune_singlenode.py      # Main tuning script
├── run.sh                  # Wrapper script
├── logs/                   # Tuning logs (timestamped)
├── OPTIMAL_CONFIGS.md      # Optimal configs reference
├── RESULTS_EP4.json        # Results for EP=4
└── README.md               # This file
```

## ⚡ Quick Start

### Use Proven Optimal Config (Recommended)

For **single-node EP=4** (4 GPUs), use these proven configs:

```python
# Add to torchtitan/distributed/deepep/utils.py line ~308-309
turbo_deepep_dispatch_tuned_config: Optional[tuple] = (24, 8, 256)
turbo_deepep_combine_tuned_config: Optional[tuple] = (24, 8, 256)
```

**Performance**: ~153-158 GB/s NVLink bandwidth (near maximum for H100/H800)

### Run Custom Tuning (Optional)

Only if you have custom hardware or need to verify:

```bash
cd scripts/deepep/torchtitan_deepep_tune
./run.sh 4 quick    # Quick: ~2 min, 6 configs
./run.sh 4 full     # Full: ~5 min, 16 configs
```

Results saved to:
- `results_ep4.json`
- `logs/tune_ep4_<timestamp>.json`

## 📊 Performance Metrics

Based on DeepEP official benchmarks (single-node H800):

| Phase | Worst Config | Optimal Config | Improvement |
|-------|--------------|----------------|-------------|
| **Dispatch** | ~165µs (~50 GB/s) | ~55µs (~153 GB/s) | **3x faster** |
| **Combine** | ~174µs (~50 GB/s) | ~58µs (~158 GB/s) | **3x faster** |

**Total MoE Layer Speedup**: 2-3x for dispatch+combine operations

## 🎯 Configuration Format

### Single-Node (Intranode)

Format: `(num_sms, nvl_chunk, nvl_buffer)`

- **num_sms**: 24 (number of SMs to use)
- **nvl_chunk**: 8 (NVLink chunk size)
- **nvl_buffer**: 256 (NVLink buffer size)

### Multi-Node (Internode)

For multi-node, see torchtitan-amd repo. Format is different: `(num_sms, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)`

## 🔧 Usage in TorchTitan

Edit `torchtitan/distributed/deepep/utils.py`:

```python
class DeepEPTokenDispatcher:
    turbo_deepep_backend: str = "deepep"
    turbo_deepep_num_cus: int = 24  # ← num_sms
    turbo_sync_free_moe: bool = False
    turbo_deepep_num_worst_tokens: int = 0
    turbo_deepep_dispatch_tuned_config: Optional[tuple] = (24, 8, 256)  # ← ADD
    turbo_deepep_combine_tuned_config: Optional[tuple] = (24, 8, 256)   # ← ADD
    use_turbo_grouped_mlp: bool = False
```

## 📝 Files

- **`tune_singlenode.py`**: Main tuning script based on DeepEP test_intranode.py
- **`run.sh`**: Wrapper to run tuning with torchrun
- **`OPTIMAL_CONFIGS.md`**: Reference document with optimal configs
- **`RESULTS_EP4.json`**: Benchmark results for EP=4
- **`logs/`**: Timestamped logs for each tuning run

## ❓ When to Tune

**DON'T tune if:**
- Using standard H100/H800 GPUs
- Single node with standard NVLink
- Default config (24, 8, 256) already optimal

**DO tune if:**
- Custom GPU hardware
- Different NVLink topology
- Debugging performance issues
- Need to squeeze every bit of performance

## 📚 References

- DeepEP GitHub: https://github.com/deepseek-ai/DeepEP
- Performance benchmarks: DeepEP README.md lines 14-23
- Test implementation: /home/phuc/workspace/moe/DeepEP/tests/test_intranode.py

## 🧹 Clean Code

- Single organized directory
- Clear file structure
- Timestamped logs
- No scattered files
- Minimal, focused implementation
