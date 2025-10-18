# PyTorch Compatibility Shim System (Experimental)

This document describes the experimental compatibility shim system that allows TorchTitan to run on both PyTorch nightly and stable releases (e.g., PyTorch 2.8.0).

## Overview

The shim system is implemented in `torchtitan/experiments/compat/compat.py` and automatically patches missing PyTorch APIs when the package is imported. This allows developers using stable PyTorch releases to use TorchTitan without requiring PyTorch nightly.

## How It Works

The compatibility system uses two approaches:

### 1. Import Hook for Missing Modules
For completely missing modules (like `torch.distributed.checkpoint._consolidate_hf_safetensors`), a custom meta path finder intercepts imports and provides shim modules with stub implementations.

### 2. Runtime Patching for Missing Classes
For existing modules that are missing specific classes (like `DefaultStager` in `torch.distributed.checkpoint.staging`), the shim system directly adds the missing classes to the existing module at import time.

## Automatic Activation

The shim system is automatically activated when you import the `torchtitan` package:

```python
import torchtitan  # Shims are installed automatically
```

This happens in `torchtitan/__init__.py`, which imports `torchtitan.experiments.compat` before anything else.

## Currently Shimmed APIs

### 1. Checkpoint Consolidation (`torch.distributed.checkpoint._consolidate_hf_safetensors`)
- `consolidate_safetensor_files` - Raises NotImplementedError
- `consolidate_safetensors_files_on_every_rank` - Raises NotImplementedError

**Note:** HuggingFace checkpoint export requires PyTorch nightly.

### 2. Checkpoint Staging (`torch.distributed.checkpoint.staging`)
- `StagingOptions` - Simple placeholder for staging configuration
- `DefaultStager` - Falls back to `BlockingAsyncStager` if available

### 3. Pipeline Schedules (`torch.distributed.pipelining.schedules`)
- `ScheduleDualPipeV` - Raises NotImplementedError if instantiated

**Note:** Use a different pipeline schedule if you hit this error.

### 4. Flex Attention (`torch.nn.attention.flex_attention`)
- `AuxOutput` - NamedTuple for auxiliary flex_attention outputs

### 5. Checkpoint Wrapper (`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`)
- Wraps `checkpoint_wrapper` function to filter out the `early_stop` parameter which is not available in PyTorch 2.8.0
- The `early_stop` parameter is silently ignored in stable PyTorch

## Adding New Shims

If you encounter a new missing API when using stable PyTorch, you can add a shim by:

1. **For missing modules:** Add a factory function to `torchtitan/experiments/compat/compat.py` and register it with `register_shim()`

```python
def _shim_new_module():
    module = ModuleType('torch.some.missing.module')
    # Add functions/classes to the module
    return module

# In install_shims():
register_shim('torch.some.missing.module', _shim_new_module)
```

2. **For missing classes in existing modules:** Add a function that patches the existing module

```python
def _shim_existing_module():
    from torch.some import existing_module

    class MissingClass:
        # Implementation or stub
        pass

    existing_module.MissingClass = MissingClass
    return existing_module

# In install_shims():
_shim_existing_module()
```

## Testing

To verify the shim system works:

```bash
# Should succeed with PyTorch 2.8.0
python -c "import torchtitan; print('Shims loaded successfully')"

# Try importing a shimmed module
python -c "from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files_on_every_rank"

# Run the test suite
python -m torchtitan.experiments.compat.test_compat
```

## Known Limitations

1. **HuggingFace Checkpoint Export:** Not supported in stable PyTorch. Set `checkpoint.last_save_in_hf = false` in your config.

2. **ScheduleDualPipeV:** Not available in stable PyTorch. Use a different pipeline schedule.

3. **Async Checkpoint Staging:** Limited functionality with the shim. Some advanced features may not work.

## Version Compatibility

- **PyTorch Nightly:** All features work natively, shims are harmless
- **PyTorch 2.8.0:** Tested and working with limitations noted above
- **Older versions:** May require additional shims

## Philosophy

The shim system follows these principles:

1. **Simple and Transparent:** Easy to understand and extend
2. **Fail-Fast:** Unsupported features raise clear errors explaining limitations
3. **Non-Intrusive:** Works automatically without code changes
4. **Compatible:** Harmless when used with PyTorch nightly

## Troubleshooting

If you encounter an import error:

1. Check if it's a PyTorch API that's missing in your version
2. Add a shim following the patterns in `torchtitan/experiments/compat/compat.py`
3. Test that both stable and nightly PyTorch work with your shim

For feature limitations, the error messages will guide you to either:
- Upgrade to PyTorch nightly
- Use an alternative feature
- Disable the feature in your configuration

## Experimental Status

This compatibility system is experimental and may change in future releases. It is designed to help users who cannot use PyTorch nightly for various reasons (e.g., stability requirements, deployment constraints).
