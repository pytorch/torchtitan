# 1. Problem 

## Text to Image - Flux.1-schnell.

[Torchtitan](https://github.com/pytorch/torchtitan) provides an implementation of the Flux model from [Black Forest Labs](https://bfl.ai/). We adapt this for MLPerf Training. The relevant files are under `torchtitan/experiments/flux`.
These files plug in to the rest of torchtitan.

```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

# 2. Directions
## Steps to configure machine
To use this repository, please ensure your system can run docker containers and has appropriate GPU support (e.g. for CUDA GPUs, please make sure the appropriate drivers are set up)

Without docker, follow the [instructions](https://github.com/pytorch/torchtitan?tab=readme-ov-file#installation) to install torchtitan and additionally install `requirements-mlperf.txt` and `torchtitan/experiments/flux/requirements.txt`.

### Container setup
To build the container:
```bash
cd torchtitan
docker build -t <tag> -f Dockerfile .
```

Before entering the container, create a directory for the models to be downloaded, and a directory to be used as huggingface cache (necessary for some operations):

```bash
mkdir <models directory>
mkdir <hf_cache_directory>
```

```bash
docker run -it --rm \
--gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
--network=host --ipc=host \
-v <hf_cache_directory>:/root/.cache \
-v <path for dataset storage>:/dataset \
-v <models directory>:/models \
<tag> bash
```

## Steps to download and verify data
For all steps below, they are assumed to run inside the container

### CC12M dataset
To download the cleaned and subsetted dataset, run the following:

**Note:** We reccomend training directly on preprocessed embeddings. To do that, skip [here](#preprocessing).

```bash
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-disk.uri
```

#### Optionally, to generate the data from scratch

Download the dataset with the following command. This requires ~1TB of storage.
```bash
HF_TRANSFER=1 huggingface-cli download --repo-type dataset pixparse/cc12m-wds --local-dir /dataset/cc12m-wds
```
Then, we remove problematic indices and keep only the first 10% of this data (rounded to 1,099,776 samples so it is nicely divisible by large powers of 2).
Depending on your CPU, you may wish to change `--num_workers` and `--batch_size`. This only impacts the runtime of this script,
the final result will be not be affected by these parameters.

```bash
python torchtitan/experiments/flux/scripts/clean_cc12m.py --input_dir /dataset/cc12m-wds --output_dir /dataset/cc12m_disk --filter_file torchtitan/experiments/flux/scripts/problematic_indices.txt --num_workers=16 --batch_size 1000
```
(Optional) Remove the downloaded dataset to reclaim space: `rm -r /dataset/cc12m-wds`

The filter file is included in this repository. It was generated using `torchtitan/experiments/flux/scripts/find_problematic_indices.py`.

### COCO-2014 subset

For validation purposes, each sample of the dataset is associated with a timestep that is used to evaluate it.
For more details, consult the [evaluation algorithm](#quality-metric)
To download the cleaned data, run the following:

**Note:** We reccomend training directly on preprocessed embeddings. To do that, skip [here](#preprocessing).

```bash
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco.uri
wget https://training.mlcommons-storage.org/flux_1/datasets/val2014_30k.tsv
```

#### Optionally, to generate the data from scratch

The number of samples is taken from the previous stable diffusion benchmark, but rounded slightly to be divisible by large powers of 2 (29,696).

1. download coco-2014 validation dataset: `DOWNLOAD_PATH=/dataset/coco2014_raw bash torchtitan/experiments/flux/scripts/coco-2014-validation-download.sh`
2. Create the validation subset, resize to 256x256 and convert to webdataset: `python torchtitan/experiments/flux/scripts/coco_to_webdataset.py --input-images-dir /dataset/coco2014_raw/val2014 --input-captions-file /dataset/coco2014_raw/annotations/captions_val2014.json --output-dir /dataset/coco --num-samples 29696 --width 256 --height 256 --samples-per-shard 1000 --output-tsv-file /dataset/val2014_30k.tsv`

##### Download the encoders
Download the autoencoder, t5 and clip models from HuggingFace. For the autoencoder, you must acquire your own access token from hf
with access rights to https://huggingface.co/black-forest-labs/FLUX.1-schnell.

**Note:** If training from preprocessed embeddings, this step is not required.

```bash
python torchtitan/experiments/flux/scripts/download_encoders.py --local_dir /models --hf_token <your_access_token>
```

### Preprocessing
Since the encoders are frozen during training, it is possible to do additional preprocessing to avoid having to repeatedly encode data on the fly.

To download this data, run the following:

```bash
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

The above requires ~2.5TB of storage.


#### Optionally, to run the preprocessing yourself

We reccomend doing this over multiple GPUs. Depending on the GPU memory, you may need to adjust the batch size.
**Due to the dataset size, using a different number of GPUs or batch size may result in hangs. Please make sure the number of samples is divisible by batch_size x NGPUs**
To do this, run:

```bash
NGPU=8 torchtitan/experiments/flux/scripts/run_preprocessing.sh --training.dataset_path=/dataset/cc12m_disk --training.dataset=cc12m_disk --eval.dataset= --training.batch_size=256 --preprocessing.output_dataset_path=/dataset/cc12m_preprocessed
```

The above may take a few hours and will require approximately 2.5TB of storage.

For the validation dataset:
```bash
NGPU=4 torchtitan/experiments/flux/scripts/run_preprocessing.sh --training.dataset=coco --training.dataset_path=/dataset/coco --eval.dataset= --training.batch_size=128 --preprocessing.output_dataset_path=/dataset/coco_preprocessed
```
Additionally, this script will generate encodings representing empty encodings which are used for guidance.

(Optional) Remove the intermediate parquet files to reclaim space: `rm -r /dataset/cc12m_preprocessed /dataset/coco_preprocessed`

To make use of the preprocessed data, switch to the config file `flux_schnell_mlperf_preprocessed.toml`.
This sets `--training.dataset=cc12m_preprocessed` and `--training.dataset_path=/dataset/cc12m_preprocessed/*`
for the training data, and `--eval.dataset=coco_preprocessed`, `--eval.dataset_path=/dataset/coco_preprocessed/*` for the eval data,
while also avoiding loading encoders with `--encoder.autoencoder_path= --encoder.t5_encoder= --encoder.clip_encoder=`.

### Steps to run and time
All steps below are assumed to be run inside the container. 

The training script uses config files to pass parameters. You can find these in `torchtitan/experiments/flux/train_configs`.
Additionally, parameters can be set or overridden in the cli.
For example, passing `--optimizer.lr=1e-3` will set the learning rate to `1e-3`.
An exhaustive list of all these parameters can be seen by running the training by running `CONFIG=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml NGPU=1 bash torchtitan/experiments/flux/run_train.sh --help` with the desired config file.

Finally, the launch scripts rely on environment variables. These are explained below.


```bash
docker run -it --rm \
--gpus all --ipc=host --ulimit memlock=-1 \
--ulimit stack=67108864 \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
-v hf_cache:/root/.cache \
-v <path for dataset storage>:/dataset/ \
-v <path for model storage>/coco:/model \
<tag> bash
```

#### Basic run
Environment variables are passed to the run script (launch script in the case of slurm).
Variables passed after are passed to torchtitan. These variables override those defined in the config file.
For a complete list of options, run the train script with `--help`.

`CONFIG=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml NGPU=<number of GPUs> bash torchtitan/experiments/flux/run_train.sh --training.batch_size=1 --training.seed=1234`

#### Longer run
**For longer runs, we expect a system with a slurm-based cluster.**

Make sure to edit the headers for the run.sub script to match the requirements of your cluster (in particular the account field).

```bash
export DATAROOT=<path_to_data>
export MODELROOT=<path_to_saved_encoders>
export LOGDIR=<output directory>
export CONFIG_FILE=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml
export CONT=<tag>
export SEED=<seed>
sbatch -N <number of nodes> -t <time> run.sub
```

`DATAROOT` should be set to the path where data resides. e.g. `${DATAROOT}/cc12m_disk` should point to the CC12M training dataset. This will be mounted under `/dataset/`.
`MODELROOT` should be set to the point where the previously downloaded encoders reside. If `SEED` is not set, a random one will be assigned.

Any additional parameters may be passed after the run.sub, and will be forwarded to the training script, overriding those in the config.
e.g. if the datasets were saved with different names from those in the instructions above, you may explicitly set the dataset paths with `--training.dataset_path=/dataset/...` and `--eval.dataset_path=`.

By default, checkpointing is disabled. You may enable it by setting the env var ENABLE_CHECKPOINTING=True. You can set the checkpointing interval.
with `--checkpoint.interval=<steps>`.

Additionally, by default, the model will run with HSDP (sharding over gpus in the same node, and using DDP across different nodes).
You may modify this by passing `--parallelism.data_parallel_replicate_degree` and `--parallelism.data_parallel_shard_degree`.

Finally, torch.compile is disabled by default. To enable it, pass `--training.compile`.

Given the substantial variability among Slurm clusters, users are encouraged to review and adapt these scripts to fit their specific cluster specifications.

In any case, the dataset and checkpoints are expected to be available to all the nodes.

# 3. Dataset/Environment
### Publication/Attribution
We use the CC12M dataset available at https://huggingface.co/datasets/pixparse/cc12m-wds

```
@inproceedings{changpinyo2021cc12m,
  title = {{Conceptual 12M}: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts},
  author = {Changpinyo, Soravit and Sharma, Piyush and Ding, Nan and Soricut, Radu},
  booktitle = {CVPR},
  year = {2021},
}
```
We use the COCO2014 dataset for validation.

```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer vision--ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### Data preprocessing
For both datasets, images are resized to 256x256 using a bicubic interpolation.

The ~10% of the CC12M dataset is used (1,099,776 samples).
The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. 
However, our benchmark uses only a subset of 29,696 images and annotations chosen at random with a preset seed.

Optionally, the training and validation datasets are preprocessed by running the encoders offline before training.

# 4. Model
### Publication/Attribution
This model largely follows the Flux.1-schnell model, as implemented by torchtitan.
In turn, the model code is largely based on the model open-sourced in [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by [Black Forest Labs](https://bfl.ai/).

```
@inproceedings{esser2024scaling,
  title={Scaling rectified flow transformers for high-resolution image synthesis},
  author={Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and Entezari, Rahim and M{\"u}ller, Jonas and Saini, Harry and Levi, Yam and Lorenz, Dominik and Sauer, Axel and Boesel, Frederic and others},
  booktitle={Forty-first international conference on machine learning},
  year={2024}
}
```

### List of layers 

| **Component** | **Architecture** | **Parameters** | **Technical Details** |
|---------------|------------------|----------------|----------------------|
| **Text Encoders (Frozen)** | | | |
| └ [VIT-L CLIP text encoder](https://huggingface.co/openai/clip-vit-large-patch14) | Transformer | ~123M | Max sequence length: 77 tokens |
| | | | Output dimension: 768 |
| └ [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl) | Transformer | ~11B | Max sequence length: 256 tokens |
| | |  | Output dimension: 4096 |
| **Image Encoder (Frozen)** | | | |
| └ [VAE (Variational AutoEncoder)](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | CNN | ~84M | Downscaling factor: 8 (256→32) |
| | | | Channel depth: 16 |
| **Diffusion Transformer** | | | |
| └ [Flux Diffusion Transformer](https://github.com/black-forest-labs/flux/) | Multimodal Diffusion Transformer (MMDiT) | ~11.9B |
| | **Double Stream Blocks** | | **19 layers** |
| | **Single Stream Blocks** | | **38 layers** |
| | | | 24 attention heads per layer | 
| | | | Hidden dimension: 3072 |
| | | | MLP ratio: 4.0 | | Processes 64 input channels |

### Loss function
The MSE calculated over latents is used for the loss
### Optimizer
AdamW
### Precision
The model runs with BF16 by default. This can be changed by setting `--training.mixed_precision_param=float32`.
### Weight initialization
The weight initialization strategy is taken from torchtitan. It consists of a mixture of constant, Xavier and Normal initialization.
For precise details, we encourage the consultation of the code at `torchtitan/experiments/flux/model/model.py:init_weights`.

# 5. Quality
### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
The validation dataset is prepared in advance so that each sample is associated with a timestep.
This is an integer from 0 to 7 inclusive, and thus should be divided by `8.0` to obtain the timestep.

The algorithm is as follows:

```pseudocode
ALGORITHM: Validation Loss Computation

INPUT:
  - validation_samples: set of validation data samples
  - num_timesteps: 8 (number of equidistant time steps)

INITIALIZE:
  - sum[8]: array of zeros for accumulating losses
  - count[8]: array of zeros for counting samples per timestep
  - t: 0 (current timestep index)

FOR each sample in validation_samples:
    loss = forward_pass(sample, timestamp=t/8)
    sum[t] += loss
    count[t] += 1
    t = (t + 1) % num_timesteps

mean_per_timestep = sum / count
validation_loss = mean(mean_per_timestep)

RETURN validation_loss
```

As we ensure that the validation set has an equal number of samples per timestep, 
a simple average of all loss values is equivalent to the above.

### Quality target
0.586
### Evaluation frequency
Every 262,144 training samples.
### Evaluation thoroughness
29,696 samples
