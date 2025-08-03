# Liger-Kernel Fused Linear Cross Entropy Loss

Liger-Kernel provides fused linear cross entropy loss that combines the final linear layer computation with cross-entropy loss calculation in a single optimized kernel. This fusion eliminates the need to store intermediate logits in memory, providing significant memory savings and performance improvements for large vocabulary models.

## Benefits

- **Memory Efficiency**: Eliminates intermediate logit storage, reducing peak memory usage
- **Performance**: Single fused kernel is faster than separate linear + cross-entropy operations  
- **Numerical Equivalence**: Produces identical results to the standard approach
- **Training Only**: Validation and generation use standard approach for simplicity

## Installation

Install Liger-Kernel via pip:
```bash
pip install liger-kernel
```

## Usage

### Configuration

Enable Liger-Kernel fused linear cross entropy loss via configuration:

**Command line:**
```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --liger_kernel.enable_fused_linear_cross_entropy
```

**TOML configuration file:**
```toml
[liger_kernel]
enable_fused_linear_cross_entropy = true
```

### Example Training Command

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_train.sh \
    --liger_kernel.enable_fused_linear_cross_entropy \
    --training.steps 1000
```

### With torch.compile

Liger-Kernel fused loss can be used with torch.compile for additional optimizations:

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_train.sh \
    --liger_kernel.enable_fused_linear_cross_entropy \
    --training.compile \
    --training.steps 1000
```

## Implementation Details

### How It Works

1. **Standard Approach**: `hidden_states → linear(hidden_states) → cross_entropy(logits, labels)`
2. **Liger Fused Approach**: `hidden_states + weight + labels → fused_linear_cross_entropy → loss`

The fused kernel directly computes the cross-entropy loss from hidden states and weights without materializing the intermediate logits tensor.

### Model Integration

The integration requires minimal changes to the model:
- The `Transformer.forward()` method gains a `return_hidden_states` parameter
- When enabled, the model returns hidden states before the final linear layer
- The training loop handles the fused loss computation externally

### Memory Layout

```python
# Input shapes for fused loss function:
# weight: [vocab_size, hidden_dim] 
# hidden_states: [batch_size, seq_len, hidden_dim]
# target: [batch_size, seq_len]
# 
# Tensors are reshaped internally:
# hidden_states: [batch_size * seq_len, hidden_dim]  
# target: [batch_size * seq_len]
```

## Compatibility

### Supported Features
- ✅ Data Parallel (FSDP)
- ✅ Tensor Parallel  
- ✅ Context Parallel
- ✅ torch.compile
- ✅ Mixed precision training
- ✅ Gradient accumulation

### Limitations
- ❌ **Pipeline Parallelism**: Not supported (validation prevents this combination)
- ❌ **Validation/Generation**: Uses standard approach (not fused)

### Error Handling

The implementation includes proper error handling:
- Graceful fallback when Liger-Kernel is not installed
- Clear error messages with installation instructions
- Validation prevents incompatible parallelism combinations

## Performance Considerations

### Memory Benefits
- Eliminates intermediate logit tensor storage: `[batch_size, seq_len, vocab_size]`
- For large vocabularies, this can save significant memory
- Memory savings scale with batch size and sequence length

### Performance Benefits
- Single fused kernel reduces kernel launch overhead
- Better memory bandwidth utilization
- Reduced memory allocations and deallocations

### When to Use
- Large vocabulary models
- Memory-constrained training scenarios  
- When training with large batch sizes or long sequences

## Troubleshooting

### Common Issues

**Import Error:**
```
ImportError: Liger-Kernel is not installed. Please install it with: pip install liger-kernel
```
Solution: Install liger-kernel package

**Pipeline Parallelism Error:**
```
RuntimeError: Liger-Kernel fused linear cross entropy loss is not compatible with Pipeline Parallelism
```
Solution: Disable either Pipeline Parallelism (`--parallelism.pipeline_parallel_degree 1`) or Liger-Kernel (`--liger_kernel.enable_fused_linear_cross_entropy False`)

### Verification

To verify that Liger-Kernel is working correctly:
1. Check training logs for "Using Liger-Kernel fused linear cross entropy loss"
2. Monitor memory usage - should see reduction in peak memory
3. Compare loss curves with standard training (should be identical)

## References

- [Liger-Kernel GitHub Repository](https://github.com/linkedin/Liger-Kernel)
- [Liger-Kernel Documentation](https://github.com/linkedin/Liger-Kernel#readme)