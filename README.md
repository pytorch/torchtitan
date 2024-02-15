# torchtrain

Note: This repository is currently under heavy development.

torchtrain contains PyTorch native parallelisms, tools and utilities to train large models.

# Installation

Install PyTorch from source or install the latest pytorch nightly, then install requirements by

```python
pip install -r requirements.txt
```

Install additional dev requirements if you want to contribute to the repo:
```
pip install -r dev-requirements.txt
```

run the llama debug model locally to verify the setup is correct:

```
./run_llama_train.sh
```

# TensorBoard

To visualize training metrics on TensorBoard:

1. (by default) set `enable_tensorboard = true` in `torchtrain/train_configs/train_config.toml`

2. set up SSH tunneling
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. then in the torchtrain repo
```
tensorboard --logdir=./torchtrain/outputs/tb
```

4. go to the URL it provides OR to http://localhost:6006/
