# torchtitan

Note: This repository is currently under heavy development.

`torchtitan` is a proof-of-concept for Large-scale LLM training using native PyTorch. It is (and will continue to be) a repo to showcase the PyTorch's latest distributed and training features in a clean, minimal codebase. torchtitan is complementary to and not a replacement for any of the great large-scale LLM training codebases such as Megatron, Megablocks, LLM Foundry, Deepspeed, Bloom, etc. Instead, we hope that the features showcased in torchtitan will be adopted by these codebases quickly. torchtitan is unlikely to ever grow a large community around it.

## Design Principles

While torchtitan utilizes the PyTorch ecosystem for things like data loading (i.e. HuggingFace datasets), the core functionality is written in PyTorch.

* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code, when applying 1D/2D or 3D Parallelisms.
* Modular components instead of monolithic codebase

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

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a .toml file or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.

## Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the multinode_trainer.slurm file to submit your sbatch job.</br>
Note that you will need to adjust the number of nodes and gpu count to your cluster configs.</br>
<b>To adjust total nodes:</b>
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```
should both be set to your total node count.
Then update the srun launch parameters to match:
```
srun torchrun --nnodes 2
```
where nnodes is your total node count, matching the sbatch node count above.

<b>To adjust gpu count per node:</b>

If your gpu count per node is not 8, adjust:

```--nproc_per_node```

 in the torchrun command and

```#SBATCH --gpus-per-task```

in the SBATCH command section.
