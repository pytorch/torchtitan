# torchtrain

Note: This repository is currently under heavy development.

torchtrain contains PyTorch native parallelisms, tools and utilities to train large models.

# Installation

install PyTorch from source or install the latest pytorch nightly, then install requirements by

```python
pip install -r requirements.txt
```

download tokenizer from HF
This part is needed first time if there's no tokenizer locally by run:

```
python torchtrain/datasets/download_tokenizer.py --hf_token your_token
```

run the llama debug model locally to verify the setup is correct:

```
./run_llama_train.sh
```
