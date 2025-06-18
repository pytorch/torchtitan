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
### Steps to configure machine
To use this repository, please ensure your system can run docker containers and has appropriate GPU support (e.g. for CUDA GPUs, please make sure the appropriate drivers are set up)

**For all instructions that follow, make sure you are in the `flux/torchtitan` directory.**

Without docker, follow the instructions to install torchtitan and additionally install `requirements-mlperf.txt` and `torchtitan/experiments/flux/requirements.txt`.

### Container setup
To build the container:
```docker build -t <tag> -f Dockerfile .```

Before entering the container, create a directory for the models to be downloaded, and a directory to be used as huggingface cache (necessary for some operations):

```bash
mkdir models
mkdir hf_cache
```

```
docker run -it --rm \
--gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
-v hf_cache:/root/.cache \
-v <path for dataset storage>:/dataset \
-v <path for model storage>:/models \
<tag> bash
```
Note: it's recommended to map your .ssh folder to inside the container, so that it's easier for the code to set up remote cluster access.

### Steps to download and verify data
For all steps below, they are assumed to run inside the container

#### CC12M dataset
Download the dataset with
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

#### COCO-2014 subset
The number of samples is taken from the previous stable diffusion benchmark, but rounded slightly to be divisible by large powers of 2.

1. download coco-2014 validation dataset: `DOWNLOAD_PATH=/dataset/coco2014_raw bash torchtitan/experiments/flux/scripts/coco-2014-validation-download.sh`
2. create the validation subset, and resize the images to 256x256: `bash torchtitan/experiments/flux/scripts/coco-2014-validation-split-resize.sh --input-images-path /dataset/coco2014_raw/val2014 --input-coco-captions /dataset/coco2014_raw/annotations/captions_val2014.json --output-images-path /dataset/coco2014 --output-tsv-file /dataset/val2014_30k.tsv --num-samples 29696`
3. convert to webdataset: `python torchtitan/experiments/flux/scripts/coco_to_webdataset.py --tsv_file /dataset/val2014_30k.tsv --image_dir /dataset/coco2014 --output_dir /dataset/coco`
4. (optional) remove intermediate datasets to reclaim space: `rm -r /dataset/coco2014_raw /dataset/coco2014`

#### Download the encoders
Download the autoencoder, t5 and clip models from HuggingFace. For the autoencoder, you must acquire your own access token from hf
with access rights to https://huggingface.co/black-forest-labs/FLUX.1-schnell.

```bash
python torchtitan/experiments/flux/scripts/download_encoders.py --local_dir /models --hf_token <your_access_token>
```
#### Preprocessing
Since the encoders are frozen during training, it is possible to do additional preprocessing to avoid having to repeatedly encode data on the fly.
To do this, run:

```bash
NGPU=8 torchtitan/experiments/flux/scripts/run_preprocessing.sh --training.dataset_path=/dataset/cc12m_disk --training.dataset=cc12m-disk --eval.dataset= --training.batch_size=256 --preprocessing.output_dataset_path=/dataset/cc12m_preprocessed
```

The above may take a few hours and will require approximately 2.5TB of storage.

For the validation dataset:
```bash
NGPU=1 torchtitan/experiments/flux/scripts/run_preprocessing.sh --training.dataset=coco --training.dataset_path=/dataset/coco --eval.dataset= --training.batch_size=128 --preprocessing.output_dataset_path=/dataset/coco_preprocessed
```
Additionally, this script will generate encodings representing empty encodings which are using for guidance.


To make use of the preprocessed data, set `--training.dataset=cc12m-preprocessed` and `--training.dataset_path=/dataset/cc12m_preprocessed/*`
for the training data, and `--eval.dataset=coco-preprocessed`, `--eval.dataset_path=/dataset/coco_preprocessed/*` for the eval data.

When using preprocessed data, we don't need to load any of the encoders. To do this, pass `--encoder.autoencoder_path= --encoder.t5_encoder= --encoder.clip_encoder=`.

Alternatively, switch to the config file `flux_schnell_mlperf_preprocessed.toml` to automatically set the above flags

### Steps to run and time
All steps below are assumed to be run inside the container. 

The first time this is executed, checkpoints for the text encoders will automatically be downloaded from HF.
To prevent this from happening every time, we encourage users to create a directory to be used as the HF cache and mount
it to the container, as below.

```
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
`CONFIG=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml NGPU=<number of GPUs> bash torchtitan/experiments/flux/run_train.sh --training.dataset=cc12m-disk --eval.dataset=coco --training.batch_size=1`

#### Longer run
**For longer runs, we expect a system with a slurm-based cluster.**

Make sure to edit the headers for the run.sub script to match the requirements of your cluster (in particular the account field).

```bash
export LOGDIR=<output directory>; export CONFIG_FILE=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml; export CONT=<tag>; export DATAROOT=<path for dataset storage>; sbatch -N <number of nodes> -t <time> run.sub <additional parameters here. e.g. --training.dataset_path=/dataset/...>
```

`DATAROOT` should be set to the path where data resides. e.g. `${DATAROOT}/cc12m_disk` should point to the CC12M training dataset. This will be mounted under `/dataset/`.

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
Without FSDP enabled, only FP32 is supported. With FSDP enabled, the default becomes BF16. This can be changed using `--training.mixed_precision_param=float32`.
### Weight initialization
The weight initialization strategy is taken from torchtitan. It consists of a mixture of constant, Xavier and Normal initialization.
For precise details, we encourage the consultation of the code at `torchtitan/experiments/flux/model/model.py:init_weights`.

# 5. Quality
### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
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

### Quality target
0.6
### Evaluation frequency
Every 614400 training samples.
### Evaluation thoroughness
29,696 samples
