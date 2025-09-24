"""
python compare_distributed_run.py --steps 5 --model-filter llama3 --flavor debugmodel --nd_parallel 2d --verbose
python compare_distributed_run.py --steps 5 --model-filter llama3 --flavor flavor --nd_parallel 2d --verbose

Methodology:
    - train on FSDP with TT (baseline)
    - train on FSDP with HF (baseline)
    - For all parallelism, train with nd-// with HF
        - If one train fails:
            - generated diff between HF FSDP (baseline) HF nd-// 
            - train the nd-// TT counterpart
                - diff between TT nd-// and HF nd-//
                - diff between TT FSDP (baseline) and HF nd-//
results/
|_ meta-llama
	|_ Llama-3.2-1B
		|_ 2D
			|_ debugmodel
				|_ baseline_hf_fsdp_4gpu.log
				|_ baseline_tt_fsdp_4gpu.log
				|_ baseline_fsdp_debugmodel_4gpu_huggingface.toml
				|_ baseline_fsdp_debugmodel_4gpu_torchtitan.toml
				|_ fsdp1_cp1_tp2_pp2_debugmodel_4gpu_huggingface/
					|_ fsdp1_cp1_tp2_pp2_debugmodel_4gpu_huggingface.toml
					|_ fsdp1_cp1_tp2_pp2_debugmodel_4gpu_torchtitan.toml
					|_ fsdp1_cp1_tp2_pp2_debugmodel_4gpu_huggingface.log
					|_ diff_hf_baseline_vs_hf_nd_parallelism.log
					|_ diff_tt_nd_parallelism_vs_hf_nd_parallelism.log
					|_ diff_tt_baseline_vs_hf_nd_parallelism.log
			|_ full
				|_ baseline_hf_fsdp_4gpu.log
				|_ baseline_tt_fsdp_4gpu.log
				|_ baseline_fsdp_full_4gpu_huggingface.toml
				|_ baseline_fsdp_full_4gpu_torchtitan.toml
				|_ fsdp1_cp1_tp2_pp2_full_4gpu_huggingface/
					|_ fsdp1_cp1_tp2_pp2_full_4gpu_huggingface.toml
					|_ fsdp1_cp1_tp2_pp2_full_4gpu_torchtitan.toml
					|_ fsdp1_cp1_tp2_pp2_full_4gpu_huggingface.log
					|_ diff_hf_baseline_vs_hf_nd_parallelism.log
					|_ diff_tt_nd_parallelism_vs_hf_nd_parallelism.log
					|_ diff_tt_baseline_vs_hf_nd_parallelism.log

"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import torch

# Configure logging with colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class LogLevel(Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TEST_PASS = "TEST_PASS"
    TEST_FAIL = "TEST_FAIL"

def log_message(level: LogLevel, message: str) -> None:
    """Log a message with appropriate color coding."""
    color_map = {
        LogLevel.INFO: Colors.BLUE,
        LogLevel.SUCCESS: Colors.GREEN,
        LogLevel.WARNING: Colors.YELLOW,
        LogLevel.ERROR: Colors.RED,
        LogLevel.TEST_PASS: Colors.GREEN,
        LogLevel.TEST_FAIL: Colors.RED,
    }
    
    prefix_map = {
        LogLevel.INFO: "[INFO]",
        LogLevel.SUCCESS: "[SUCCESS]",
        LogLevel.WARNING: "[WARNING]",
        LogLevel.ERROR: "[ERROR]",
        LogLevel.TEST_PASS: "✅ TEST PASS",
        LogLevel.TEST_FAIL: "❌ TEST FAIL",
    }
    
    color = color_map[level]
    prefix = prefix_map[level]
    print(f"{color}{prefix}{Colors.NC} {message}")

@dataclass
class ParallelismConfig:
    """Configuration for a parallelism setup."""
    name: str
    dp_replicate: int
    dp_shard: int
    tp: int
    pp: int
    pp_schedule: str
    cp: int
    ep: int
    eptp: int

@dataclass
class TrainingMetrics:
    """Training metrics extracted from logs."""
    steps: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    memory: List[float] = field(default_factory=list)
    tps: List[int] = field(default_factory=list)
    tflops: List[float] = field(default_factory=list)
    mfu: List[float] = field(default_factory=list)

class CompareDistributedRun:
    """Main class for running distributed parallelism comparison tests."""
    
    # Default values
    DEFAULT_STEPS = 10
    DEFAULT_SEED = 42
    DEFAULT_FLAVOR = "debugmodel"
    # value chosen based on diff of llama3 1GPU
    DEFAULT_LOSS_ATOL = 0.02
    DEFAULT_LOSS_RTOL = 1e-5
    DEFAULT_GRAD_NORM_ATOL = 0.005
    DEFAULT_GRAD_NORM_RTOL = 1e-5
    
    MODEL_LISTS = {
        "torchtitan":  ["llama3", "deepseek_v3"],
        "huggingface": ["meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-V3"]
    }
    
    MODEL_FLAVORS = {
        "llama3": ["debugmodel", "medium", "full"],
        "deepseek_v3": ["debugmodel"],
        "meta-llama/Llama-3.2-1B": ["debugmodel", "medium", "full"],
        "deepseek-ai/DeepSeek-V3": ["debugmodel"],
    }

    #TODO(3outeille): handle slurm later for 4D/5D. Might need to rethink the whole script for that
    # Available ND parallelisms <-> number of GPUs
    ND_PARALLEL_TO_NB_GPUS = {
        "0d": 1,
        "1d": 2,
        "2d": 4,
        "3d": 8,
        "4d": 16,
        "5d": 32,
    }
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.torchtitan_root = self.script_dir.parent.parent
        self.base_results_dir = self.script_dir / "results"
        
        # Configuration parameters
        self.nd_parallel_to_nb_gpus = self.ND_PARALLEL_TO_NB_GPUS
        self.steps = self.DEFAULT_STEPS
        self.seed = self.DEFAULT_SEED
        self.model_filter = ""
        self.flavor = self.DEFAULT_FLAVOR
        self.verbose = False
        self.use_slurm = False
        self.slurm_options = []
        self.loss_atol = self.DEFAULT_LOSS_ATOL
        self.loss_rtol = self.DEFAULT_LOSS_RTOL
        self.grad_norm_atol = self.DEFAULT_GRAD_NORM_ATOL
        self.grad_norm_rtol = self.DEFAULT_GRAD_NORM_RTOL
        self.parallelism_configs: List[ParallelismConfig] = []
        self.results_dir: Optional[Path] = None

    def generate_parallelism_configs(self) -> None:
        """Generate parallelism configurations based on the number of GPUs."""
        ngpu = self.nd_parallel_to_nb_gpus[self.nd_parallel]
        configs = []

        def _get_factors(n: int) -> List[int]:
            factors = set()
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.add(i)
                    factors.add(n // i)
            return sorted(list(factors))

        # Baseline FSDP
        configs.append(ParallelismConfig(name="fsdp", dp_replicate=1, dp_shard=ngpu, tp=1, pp=1, pp_schedule="Interleaved1F1B", cp=1, ep=1, eptp=1))

        #NOTE(3outeille): No need to handle DDP (dp_replicate) as DDP is not supported > 1D parallelism"
        #(cf https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/infra/parallelize.py#L139)
        possible_fsdp = _get_factors(ngpu) # dp_shard
        possible_cp = _get_factors(ngpu)
        possible_tp = _get_factors(ngpu)
        possible_pp = _get_factors(ngpu)

        #TODO(3outeille): handle HSDP later

        for dp_shard in possible_fsdp:
            for cp in possible_cp:
                for tp in possible_tp:
                    for pp in possible_pp:
                        
                        if dp_shard * cp * tp * pp != ngpu:
                            continue

                        num_parallelisms_used = sum(parallel_degree > 1 for parallel_degree in [dp_shard, cp, tp, pp])
                        ndims_required = int(self.nd_parallel[0])
                        #NOTE(3outeille): if 2D//, we need at least 2 parallelisms to be active (> 1). For 3D //, least 3 parallelisms > 1 etc.
                        if ndims_required > 1 and num_parallelisms_used < ndims_required:
                            continue

                        configs.append(
                            ParallelismConfig(
                                name=f"fsdp{dp_shard}_cp{cp}_tp{tp}_pp{pp}",
                                dp_replicate=1,
                                dp_shard=dp_shard,
                                tp=tp,
                                pp=pp,
                                pp_schedule="Interleaved1F1B",
                                cp=cp,
                                ep=1,
                                eptp=1
                            )
                        )

                        # NOTE(3outeille): EP borrowing degree from dp_shard
                        configs.append(
                            ParallelismConfig(
                                name=f"fsdp{dp_shard}_cp{cp}_tp{tp}_pp{pp}_ep{dp_shard}",
                                dp_replicate=1,
                                dp_shard=dp_shard,
                                tp=tp,
                                pp=pp,
                                pp_schedule="Interleaved1F1B",
                                cp=cp,
                                ep=dp_shard,
                                eptp=1
                            )
                        )
        
    
        # Remove duplicates and assign to instance
        unique_configs = []
        seen_configs = set()
        for config in configs:
            # Create a tuple of the config values to check for duplicates
            config_tuple = (config.dp_replicate, config.dp_shard, config.tp, config.pp, config.cp, config.ep, config.eptp)
            if config_tuple not in seen_configs:
                unique_configs.append(config)
                seen_configs.add(config_tuple)

        self.parallelism_configs = unique_configs
        
        log_message(LogLevel.INFO, f"Generated {len(self.parallelism_configs)} parallelism configurations for {ngpu} GPUs.")
        if self.verbose:
            for config in self.parallelism_configs:
                log_message(LogLevel.INFO, f"  - {config.name}: dp_replicate={config.dp_replicate}, dp_shard={config.dp_shard}, tp={config.tp}, pp={config.pp}, cp={config.cp}, ep={config.ep}, eptp={config.eptp}")
    
    def generate_config(self, config_dir: Path, config: ParallelismConfig, model_name: str, backend: str, filename: Optional[str] = None) -> Path:
        """Generate configuration file for a parallelism setup."""
        import toml

        if filename:
            config_file = config_dir / filename
        else:
            config_file = config_dir / f"{config.name}_{self.flavor}_{self.nd_parallel_to_nb_gpus[self.nd_parallel]}gpu_{backend}.toml"

        base_config = self.script_dir / "configs" / "test_template.toml"
        shutil.copy2(base_config, config_file)

        # Load the TOML file as a dict
        with open(config_file, 'r') as f:
            config_data = toml.load(f)

        # Update [model] section
        if "model" not in config_data:
            config_data["model"] = {}
        config_data["model"]["name"] = model_name
        config_data["model"]["flavor"] = self.flavor

        # Validate flavor for model type
        if model_name in self.MODEL_FLAVORS:
            if self.flavor not in self.MODEL_FLAVORS[model_name]:
                log_message(LogLevel.WARNING, 
                           f"Flavor '{self.flavor}' not available for {model_name}. "
                           f"Available: {self.MODEL_FLAVORS[model_name]}")

        # Update [training] section
        if "training" not in config_data:
            config_data["training"] = {}
        config_data["training"]["steps"] = self.steps
        config_data["training"]["seed"] = self.seed

        # Update [parallelism] section
        if "parallelism" not in config_data:
            config_data["parallelism"] = {}
        config_data["parallelism"]["data_parallel_replicate_degree"] = config.dp_replicate
        config_data["parallelism"]["data_parallel_shard_degree"] = config.dp_shard
        config_data["parallelism"]["tensor_parallel_degree"] = config.tp
        config_data["parallelism"]["pipeline_parallel_degree"] = config.pp
        config_data["parallelism"]["pipeline_parallel_schedule"] = config.pp_schedule
        config_data["parallelism"]["context_parallel_degree"] = config.cp
        config_data["parallelism"]["expert_parallel_degree"] = config.ep
        config_data["parallelism"]["expert_tensor_parallel_degree"] = config.eptp

        # Write back the modified TOML
        with open(config_file, 'w') as f:
            toml.dump(config_data, f)

        log_message(LogLevel.INFO, f"Created config file: {config_file} for config '{config.name}' (model: {model_name})")
        return config_file
    
    def extract_metrics(self, log_file: Path) -> TrainingMetrics:
        """Extract metrics from log file."""
        metrics = TrainingMetrics()
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Regex to capture all metrics from a log line, ignoring ANSI color codes
            pattern = re.compile(
                r"step:\s*(\d+)\s*"
                r".*?loss:\s*([0-9]+\.?[0-9]*)\s*"
                r".*?grad_norm:\s*([0-9]+\.?[0-9]*)\s*"
            )

            for match in pattern.finditer(content):
                metrics.steps.append(int(match.group(1)))
                metrics.loss.append(float(match.group(2)))
                metrics.grad_norm.append(float(match.group(3)))
                
        except Exception as e:
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}: {e}")
        
        if not metrics.loss or not metrics.grad_norm:
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}")
        
        return metrics
    
    def compare_metrics(self, baseline_metrics: TrainingMetrics, test_metrics: TrainingMetrics, 
                       config_name: str) -> bool:
        """Compare metrics between baseline and test configuration."""
        if not baseline_metrics.loss or not test_metrics.loss:
            log_message(LogLevel.TEST_FAIL, f"{config_name} - Unable to extract metrics")
            return False
        
        # Convert to tensors
        baseline_loss = torch.tensor(baseline_metrics.loss)
        test_loss = torch.tensor(test_metrics.loss)
        baseline_grad_norm = torch.tensor(baseline_metrics.grad_norm)
        test_grad_norm = torch.tensor(test_metrics.grad_norm)
        
        # Check if tensors are close
        loss_pass = torch.allclose(baseline_loss, test_loss, atol=self.loss_atol, rtol=self.loss_rtol)
        grad_pass = torch.allclose(baseline_grad_norm, test_grad_norm, atol=self.grad_norm_atol, rtol=self.grad_norm_rtol)

        # Calculate max absolute differences for logging
        loss_diff = torch.max(torch.abs(baseline_loss - test_loss)).item() if baseline_loss.numel() > 0 and test_loss.numel() > 0 else 0.0
        grad_norm_diff = torch.max(torch.abs(baseline_grad_norm - test_grad_norm)).item() if baseline_grad_norm.numel() > 0 and test_grad_norm.numel() > 0 else 0.0
        
        if loss_pass and grad_pass:
            log_message(LogLevel.TEST_PASS, 
                       f"{config_name} - Max loss diff: {loss_diff:.2e}, "
                       f"Max grad norm diff: {grad_norm_diff:.2e}")
            return True
        else:
            log_message(LogLevel.TEST_FAIL,
                       f"{config_name} - Max loss diff: {loss_diff:.2e}, "
                       f"Max grad norm diff: {grad_norm_diff:.2e}")
            return False
    
    def generate_diff(self, baseline_log: Path, test_log: Path, diff_file: Path) -> None:
        """Generate diff between baseline and test logs."""
        
        def _filter_log(log_file: Path) -> Path:
            """Filter log file to normalize volatile information."""
            filtered_file = log_file.with_suffix(log_file.suffix + '.filtered')
            
            with open(log_file, 'r') as infile, open(filtered_file, 'w') as outfile:
                for line in infile:
                    # Apply filtering patterns
                    line = re.sub(r'([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?', 
                                'TIMESTAMP', line)
                    line = re.sub(r'torchrun.*--master_port[= ]([0-9]+)', 
                                'torchrun ... --master_port=XXXX', line)
                    line = re.sub(r'PID [0-9]+', 'PID XXXX', line)
                    line = re.sub(r'localhost:[0-9]+', 'localhost:XXXX', line)
                    line = re.sub(r'memory: [0-9]+\.[0-9]+GiB', 'memory: XX.XXGiB', line)
                    line = re.sub(r'tps: [0-9,]+', 'tps: XXXXX', line)
                    line = re.sub(r'tflops: [0-9]+\.[0-9]+', 'tflops: XX.XX', line)
                    line = re.sub(r'mfu: [0-9]+\.[0-9]+%', 'mfu: XX.XX%', line)
                    outfile.write(line)
            
            return filtered_file
        try:
            # Filter logs to remove timestamps and volatile information
            baseline_filtered = _filter_log(baseline_log)
            test_filtered = _filter_log(test_log)
            
            # Generate colored diff using git diff
            cmd = ["git", "diff", "--no-index", "--color=always", "--word-diff=color",
                   str(baseline_filtered), str(test_filtered)]
            
            with open(diff_file, 'w') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
            
            # Clean up filtered files
            baseline_filtered.unlink()
            test_filtered.unlink()
            
        except Exception as e:
            log_message(LogLevel.WARNING, f"Could not generate diff: {e}")
    
    def run_training(self, config_file: Path, log_file: Path, config_name: str, model_name: str) -> Optional[subprocess.CalledProcessError]:
        """Run training with given configuration."""
        log_message(LogLevel.INFO, f"Running training: {config_name} with model {model_name}")
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.ngpu}",
            "--rdzv_backend", "c10d",
            "--rdzv_endpoint=localhost:0",
            "--local-ranks-filter", "0",
            "--role", "rank",
            "--tee", "3",
            "-m", "torchtitan.train",
            "--training.seed", str(self.seed),
            "--training.deterministic",
            "--job.config_file", str(config_file)
        ]
        env = os.environ.copy()
        env["SEED"] = str(self.seed)
        env["MODEL_TYPE"] = model_name
        
        if self.verbose:
            log_message(LogLevel.INFO, f"Command: {' '.join(cmd)}")
        
        try:
            # Capture output to include it in the exception, while still writing to log file
            result = subprocess.run(
                cmd,
                cwd=self.torchtitan_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,  # decodes stdout/stderr as text
                env=env,
                check=True
            )
            with open(log_file, 'w') as f:
                f.write(result.stdout)
            
            if self.verbose:
                log_message(LogLevel.SUCCESS, f"Training completed: {config_name}")
            return None
            
        except subprocess.CalledProcessError as e:
            log_message(LogLevel.ERROR, f"Training failed: {config_name}")
            
            # Write the failed output to the log file
            with open(log_file, 'w') as f:
                if e.stdout:
                    f.write(e.stdout)

            e.add_note(f"\n--- Full output from failed process ---\n{e.stdout or '<no output captured>'}")
            return e
    
    def _compare_one_parallelism_config(
        self,
        config: "ParallelismConfig",
        hf_model_name: str,
        tt_model_name: str,
        hf_baseline_metrics: "TrainingMetrics",
        baseline_log_hf: Path,
        baseline_log_tt: Path,
    ) -> bool:
        """Compares a single parallelism configuration against the baseline."""
        # Create a subdirectory for each test configuration
        test_dir_name = f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface"
        test_dir = self.results_dir / test_dir_name
        test_dir.mkdir(exist_ok=True)

        config_filename_hf = f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.toml"
        config_file_hf = self.generate_config(config_dir=test_dir, config=config, model_name=hf_model_name, backend="huggingface", filename=config_filename_hf)
        log_path_hf = test_dir / f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.log"

        hf_run_error = self.run_training(config_file=config_file_hf, log_file=log_path_hf, config_name=config.name, model_name=hf_model_name)
        successful_hf_run = hf_run_error is None

        # Compare metrics between baseline (HF) and current (HF) nd-parallelism run
        hf_metrics = self.extract_metrics(log_path_hf)
        successful_hf_extract = self.compare_metrics(hf_baseline_metrics, hf_metrics, f"{config.name} (huggingface)")

        if successful_hf_run and successful_hf_extract:
            return True
        else:
            # Generate diff with baseline (HF)
            diff_hf_baseline_vs_hf_nd_parallelism = test_dir / "diff_hf_baseline_vs_hf_nd_parallelism.log"
            self.generate_diff(baseline_log_hf, log_path_hf, diff_hf_baseline_vs_hf_nd_parallelism)
            log_message(LogLevel.INFO, f"Diff between baseline (HF) and current (HF) nd-parallelism run saved to: {diff_hf_baseline_vs_hf_nd_parallelism}")

            # Run TT counterpart and generated diff between nd-paralellism TT and current hf nd-parallelism run
            config_filename_tt = f"{config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.toml"
            config_file_tt = self.generate_config(config_dir=test_dir, config=config, model_name=tt_model_name, backend="torchtitan", filename=config_filename_tt)
            log_path_tt = test_dir / f"{config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.log"
            tt_run_error = self.run_training(config_file=config_file_tt, log_file=log_path_tt, config_name=config.name, model_name=tt_model_name)
            if tt_run_error:
                raise ValueError(f"TorchTitan training failed for {tt_model_name}") from tt_run_error

            # generated diff between nd-paralellism TT and current hf nd-parallelism run
            diff_file_tt_nd_parallelism_vs_hf_nd_parallelism = test_dir / "diff_tt_nd_parallelism_vs_hf_nd_parallelism.log"
            self.generate_diff(log_path_tt, log_path_hf, diff_file_tt_nd_parallelism_vs_hf_nd_parallelism)
            log_message(LogLevel.INFO, f"Diff between nd-paralellism TT and current (HF) nd-parallelism run saved to: {diff_file_tt_nd_parallelism_vs_hf_nd_parallelism}")

            # generated diff between baseline TT and current hf nd-parallelism run
            diff_file_tt_baseline_vs_hf_nd_parallelism = test_dir / "diff_tt_baseline_vs_hf_nd_parallelism.log"
            self.generate_diff(baseline_log_tt, log_path_hf, diff_file_tt_baseline_vs_hf_nd_parallelism)
            log_message(LogLevel.INFO, f"Diff between baseline TT and current (HF) nd-parallelism run saved to: {diff_file_tt_baseline_vs_hf_nd_parallelism}")
            return False

    def run(self) -> int:
        """Main execution function. Runs all test suites for all models."""
        parser = argparse.ArgumentParser(
            description="Test different parallelism configurations against a baseline FSDP model.",
        )
        parser.add_argument("-m", "--model-filter", default="",
                          help="Filter models by name pattern (e.g., 'llama3')")
        parser.add_argument("-nd", "--nd_parallel", type=str, default="2d",
                          help=f"Parallelism to use (default: {self.ND_PARALLEL_TO_NB_GPUS.keys()})")
        parser.add_argument("-s", "--steps", type=int, default=self.DEFAULT_STEPS,
                          help=f"Training steps (default: {self.DEFAULT_STEPS})")
        parser.add_argument("--flavor", default=self.DEFAULT_FLAVOR,
                          help=f"Model flavor/size (default: {self.DEFAULT_FLAVOR}). "
                               f"Available: llama3=[debugmodel, medium, full], deepseek_v3=[debugmodel]")
        parser.add_argument("-v", "--verbose", action="store_true",
                          help="Verbose output")
        parser.add_argument("--loss-atol", type=float, default=self.DEFAULT_LOSS_ATOL,
                          help=f"Absolute tolerance for loss comparison (default: {self.DEFAULT_LOSS_ATOL})")
        parser.add_argument("--loss-rtol", type=float, default=self.DEFAULT_LOSS_RTOL,
                          help=f"Relative tolerance for loss comparison (default: {self.DEFAULT_LOSS_RTOL})")
        parser.add_argument("--grad-norm-atol", type=float, default=self.DEFAULT_GRAD_NORM_ATOL,
                          help=f"Absolute tolerance for grad norm comparison (default: {self.DEFAULT_GRAD_NORM_ATOL})")
        parser.add_argument("--grad-norm-rtol", type=float, default=self.DEFAULT_GRAD_NORM_RTOL,
                          help=f"Relative tolerance for grad norm comparison (default: {self.DEFAULT_GRAD_NORM_RTOL})")
        
        args = parser.parse_args()
        
        self.nd_parallel = args.nd_parallel
        self.ngpu = self.nd_parallel_to_nb_gpus[self.nd_parallel]
        self.steps = args.steps
        self.model_filter = args.model_filter
        self.flavor = args.flavor
        self.verbose = args.verbose
        self.loss_atol = args.loss_atol
        self.loss_rtol = args.loss_rtol
        self.grad_norm_atol = args.grad_norm_atol
        self.grad_norm_rtol = args.grad_norm_rtol
        
        log_message(LogLevel.INFO, "=== Distributed Parallelism Comparison ===")
        log_message(LogLevel.INFO, f"GPUs: {self.ngpu}")
        log_message(LogLevel.INFO, f"Steps: {self.steps}")
        log_message(LogLevel.INFO, f"Seed: {self.seed}")
        log_message(LogLevel.INFO, f"Model filter: {self.model_filter or 'all'}")
        log_message(LogLevel.INFO, f"Model flavor: {self.flavor}")
        print()
        
        self.base_results_dir.mkdir(exist_ok=True)

        self.generate_parallelism_configs()
        
        #TODO(3outeille): make it more generic later
        if self.model_filter == "llama3":
            hf_model_name = "meta-llama/Llama-3.2-1B"
            tt_model_name = "llama3"
        elif self.model_filter == "deepseek_v3":
            hf_model_name = "deepseek-ai/DeepSeek-V3"
            tt_model_name = "deepseek_v3"
        else:
            raise ValueError(f"Model filter {self.model_filter} not supported")
            
        model_owner, model_repo = hf_model_name.split("/", 1)
        nd_parallel_upper = self.nd_parallel.upper()
        self.results_dir = self.base_results_dir / model_owner / model_repo / nd_parallel_upper / self.flavor
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            log_message(LogLevel.INFO, f"Results directory: {self.results_dir}")

        log_message(LogLevel.INFO, "--- Running baseline (FSDP) for huggingface backend ---")

        log_message(LogLevel.INFO, f"Testing model {hf_model_name} (HF) for {self.nd_parallel} parallelism")

        baseline_config = next((c for c in self.parallelism_configs if c.name == "fsdp"), None)
        
        baseline_config_filename_hf = f"baseline_{baseline_config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.toml"
        baseline_config_file_hf = self.generate_config(config_dir=self.results_dir, config=baseline_config, model_name=hf_model_name, backend="huggingface", filename=baseline_config_filename_hf)
        baseline_log_hf = self.results_dir / f"baseline_hf_{baseline_config.name}_{self.ngpu}gpu.log"
        hf_baseline_run_error = self.run_training(config_file=baseline_config_file_hf, log_file=baseline_log_hf, config_name=baseline_config.name, model_name=hf_model_name)
        if hf_baseline_run_error:
            raise ValueError(f"Huggingface baseline (FSDP) training failed for {hf_model_name}") from hf_baseline_run_error

        hf_baseline_metrics = self.extract_metrics(baseline_log_hf)
        if not hf_baseline_metrics.loss or not hf_baseline_metrics.grad_norm:
            raise ValueError(f"Could not extract huggingface baseline metrics for {hf_model_name}")
        
        log_message(LogLevel.INFO, "--- Running baseline (FSDP) for torchtitan backend ---")

        log_message(LogLevel.INFO, f"Testing model {hf_model_name} (TT) for {self.nd_parallel} parallelism")

        baseline_config_filename_tt = f"baseline_{baseline_config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.toml"
        baseline_config_file_tt = self.generate_config(config_dir=self.results_dir, config=baseline_config, model_name=tt_model_name, backend="torchtitan", filename=baseline_config_filename_tt)
        baseline_log_tt = self.results_dir / f"baseline_tt_{baseline_config.name}_{self.ngpu}gpu.log"
        tt_baseline_run_error = self.run_training(config_file=baseline_config_file_tt, log_file=baseline_log_tt, config_name=baseline_config.name, model_name=tt_model_name)
        if tt_baseline_run_error:
            raise ValueError(f"TorchTitan baseline (FSDP) training failed for {tt_model_name}") from tt_baseline_run_error

        tt_baseline_metrics = self.extract_metrics(baseline_log_tt)
        if not tt_baseline_metrics.loss or not tt_baseline_metrics.grad_norm:
            raise ValueError(f"Could not extract TorchTitan baseline metrics for {tt_model_name}")
        
        if not self.compare_metrics(tt_baseline_metrics, hf_baseline_metrics, "baseline (TT) vs baseline (HF)"):
            raise ValueError(f"Baseline (TT) vs baseline (HF) metrics comparison failed for {tt_model_name}")

        log_message(LogLevel.INFO, "--- Comparing other parallelism configurations (huggingface) ---")
        passed_tests = 0
        failed_tests = 0
        test_configs = [c for c in self.parallelism_configs if c.name != "fsdp"]
        total_tests = len(test_configs)

        for config in test_configs:
            passed = self._compare_one_parallelism_config(
                config,
                hf_model_name,
                tt_model_name,
                hf_baseline_metrics,
                baseline_log_hf,
                baseline_log_tt,
            )
            if passed:
                passed_tests += 1
            else:
                failed_tests += 1

        print()
        
        log_message(LogLevel.INFO, "=== FINAL SUMMARY ===")
        if passed_tests == total_tests:
            log_message(LogLevel.SUCCESS, "All model tests passed! 🎉")
            return 0
        else:
            log_message(LogLevel.TEST_FAIL, f"{failed_tests} model(s) had test failures")
            log_message(LogLevel.INFO, f"Check the diff files in {self.results_dir} for details")
            return 1

def main():
    """Entry point for the script."""
    runner = CompareDistributedRun()
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())
