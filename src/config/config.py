"""Pydantic configuration models for MoE training."""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    n_layers: int = 27
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    num_experts: int = 64
    top_k: int = 6
    num_shared_experts: int = 2
    ffn_dim: int = 1408
    vocab_size: int = 151936
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    n_dense_layers: int = 0  # 0 = all layers use MoE
    depth_init: bool = True
    force_load_balance: bool = False  # debug: round-robin routing for EP correctness checks


class TrainingConfig(BaseModel):
    seq_len: int = 8192
    local_batch_size: int = 4
    grad_accum_steps: int = 16
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    max_norm: float = 1.0
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"
    seed: int = 42
    gc_freq: int = 50
    dataset: str = "c4"
    dataset_path: str | None = None
    log_freq: int = 10


class ParallelismConfig(BaseModel):
    dp_shard: int = -1  # -1 = auto (use all remaining GPUs)
    ep: int = 8


class DataConfig(BaseModel):
    tokenizer: str = "Qwen/Qwen2-7B"  # HF model ID or local path


class CheckpointConfig(BaseModel):
    enable: bool = True
    interval: int = 500
    folder: str = "checkpoint"
    async_mode: str = "disabled"
    keep_latest_k: int = 10


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    parallelism: ParallelismConfig = ParallelismConfig()
    data: DataConfig = DataConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    dump_folder: str = "./outputs"


def build_job_config(cfg: Config):
    """Translate pydantic Config into TorchTitan's JobConfig for components that need it."""
    from src.config.job_config import JobConfig

    jc = JobConfig()

    # Job
    jc.job.dump_folder = cfg.dump_folder

    # Model
    jc.model.hf_assets_path = cfg.data.tokenizer

    # Training
    jc.training.dataset = cfg.training.dataset
    jc.training.dataset_path = cfg.training.dataset_path
    jc.training.local_batch_size = cfg.training.local_batch_size
    jc.training.seq_len = cfg.training.seq_len
    jc.training.max_norm = cfg.training.max_norm
    jc.training.steps = cfg.training.max_steps
    jc.training.mixed_precision_param = cfg.training.mixed_precision_param  # type: ignore
    jc.training.mixed_precision_reduce = cfg.training.mixed_precision_reduce  # type: ignore
    jc.training.gc_freq = cfg.training.gc_freq

    # Optimizer
    jc.optimizer.lr = cfg.training.lr
    jc.optimizer.beta1 = cfg.training.beta1
    jc.optimizer.beta2 = cfg.training.beta2
    jc.optimizer.eps = cfg.training.eps
    jc.optimizer.weight_decay = cfg.training.weight_decay

    # LR Scheduler
    jc.lr_scheduler.warmup_steps = cfg.training.warmup_steps
    jc.lr_scheduler.decay_type = "cosine"
    jc.lr_scheduler.min_lr_factor = cfg.training.min_lr_ratio

    # Parallelism
    jc.parallelism.data_parallel_shard_degree = cfg.parallelism.dp_shard
    jc.parallelism.expert_parallel_degree = cfg.parallelism.ep

    # Checkpoint
    jc.checkpoint.enable = cfg.checkpoint.enable
    jc.checkpoint.interval = cfg.checkpoint.interval
    jc.checkpoint.folder = cfg.checkpoint.folder
    jc.checkpoint.async_mode = cfg.checkpoint.async_mode  # type: ignore
    jc.checkpoint.keep_latest_k = cfg.checkpoint.keep_latest_k

    # Metrics
    jc.metrics.log_freq = cfg.training.log_freq

    # Debug
    jc.debug.seed = cfg.training.seed

    # Activation checkpoint — disabled
    jc.activation_checkpoint.mode = "none"

    return jc
