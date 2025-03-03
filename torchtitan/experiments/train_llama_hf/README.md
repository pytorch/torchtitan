# Training LLAMA with HF weights

This directory contains scripts and configs for training LLAMA with HF weights using TorchTitan.

## Usage

### Install extra dependencies

```bash
pip install -r extra_requirements.txt
```

### Test loading HF weights

```bash
pytest test_loading_hf_weights.py
```

### Run training

```bash
LOG_RANK=7 bash run_train.sh
```
