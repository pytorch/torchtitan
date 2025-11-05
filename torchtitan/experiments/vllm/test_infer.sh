#!/bin/bash
# Simple test script to demonstrate vLLM inference with TorchTitan Qwen3 model

set -e

echo "========================================"
echo "TorchTitan Qwen3 + vLLM Inference Test"
echo "========================================"

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/torchtitan/checkpoint"
    echo ""
    echo "The checkpoint directory should contain:"
    echo "  - config.json (HuggingFace-style model config)"
    echo "  - Model weights (PyTorch checkpoint or safetensors)"
    exit 1
fi

MODEL_PATH="$1"

# Verify model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Verify config.json exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in $MODEL_PATH"
    echo ""
    echo "Please ensure your checkpoint contains a HuggingFace-style config.json"
    echo "See example_config.json for reference"
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo ""

# Run inference with a simple prompt
echo "Running inference with single prompt..."
python torchtitan/experiments/vllm/infer.py \
    --model-path "$MODEL_PATH" \
    --prompt "What is the meaning of life?" \
    --max-tokens 50 \
    --temperature 0.7

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
