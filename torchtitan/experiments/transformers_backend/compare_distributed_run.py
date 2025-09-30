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
                - diff between TT FSDP (baseline) and TF nd-//
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
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


console = Console()


class LogLevel(Enum):
    COMMAND = "COMMAND"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TEST_PASS = "TEST_PASS"
    TEST_FAIL = "TEST_FAIL"


def log_message(level: LogLevel, message: str, indent: int = 0, dim: bool = False) -> None:
    """Log a message with appropriate color coding."""
    style_map = {
        LogLevel.COMMAND: "dim",
        LogLevel.INFO: "blue",
        LogLevel.SUCCESS: "green",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "bold red",
        LogLevel.TEST_PASS: "green",
        LogLevel.TEST_FAIL: "bold red",
    }

    prefix_map = {
        LogLevel.COMMAND: "[COMMAND]",
        LogLevel.INFO: "[INFO]",
        LogLevel.SUCCESS: "[SUCCESS]",
        LogLevel.WARNING: "[WARNING]",
        LogLevel.ERROR: "[ERROR]",
        LogLevel.TEST_PASS: "✅ TEST PASS",
        LogLevel.TEST_FAIL: "❌ TEST FAIL",
    }

    style = style_map[level]
    prefix = prefix_map[level]
    if indent > 0:
        indent_str = "  " * (indent - 1) + "└─ "
    else:
        indent_str = ""
         
    output = ""
    if level == LogLevel.COMMAND:
        output = f"{indent_str}[{style}]{prefix} {message}[/]"
    else:
        output = f"{indent_str}[{style}]{prefix}[/] {message}"

    if dim:
        console.print(f"[dim]{output}[/dim]")
    else:
        console.print(output)


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
    DEFAULT_GRAD_NORM_ATOL = 0.02
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
        self.test_filter = ""

    def generate_parallelism_configs(self, hf_model_name: str) -> None:
        """Generate parallelism configurations based on the number of GPUs."""
        from transformers import AutoConfig

        try:
            model_config = AutoConfig.from_pretrained(hf_model_name)
            is_moe = getattr(model_config, "num_local_experts", 0) > 1
        except Exception:
            # Fallback for models not on Hub or other errors
            is_moe = False
            log_message(LogLevel.WARNING, f"Could not determine if {hf_model_name} is a MoE model from HuggingFace Hub. EP configurations will not be generated.")

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
        configs.append(ParallelismConfig(name="fsdp", dp_replicate=1, dp_shard=ngpu, tp=1, pp=1, pp_schedule="1F1B", cp=1, ep=1, eptp=1))

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
                                pp_schedule="1F1B",
                                cp=cp,
                                ep=1,
                                eptp=1
                            )
                        )

                        if is_moe:
                            # NOTE(3outeille): EP borrowing degree from dp_shard
                            configs.append(
                                ParallelismConfig(
                                    name=f"fsdp{dp_shard}_cp{cp}_tp{tp}_pp{pp}_ep{dp_shard}",
                                    dp_replicate=1,
                                    dp_shard=dp_shard,
                                    tp=tp,
                                    pp=pp,
                                    pp_schedule="1F1B",
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
        
        log_message(
            LogLevel.INFO,
            f"Generated {len(self.parallelism_configs)} parallelism configurations for {ngpu} GPUs.",
        )
        configs_to_display = self.parallelism_configs
        table_title = "[bold]Generated Parallelism Configurations[/bold]"

        if self.test_filter:
            # Keep fsdp baseline and anything that matches the filter
            configs_to_display = [c for c in self.parallelism_configs if c.name == "fsdp" or self.test_filter in c.name]
            table_title = f"[bold]Filtered Parallelism Configurations (filter: [cyan]'{self.test_filter}'[/cyan])[/bold]"

        table = Table(
            title=table_title,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("dp_replicate", justify="right")
        table.add_column("dp_shard", justify="right")
        table.add_column("tp", justify="right")
        table.add_column("pp", justify="right")
        table.add_column("cp", justify="right")
        table.add_column("ep", justify="right")
        table.add_column("eptp", justify="right")

        for config in configs_to_display:
            table.add_row(
                config.name,
                str(config.dp_replicate),
                str(config.dp_shard),
                str(config.tp),
                str(config.pp),
                str(config.cp),
                str(config.ep),
                str(config.eptp),
            )
        console.print(table)
        console.print()
    
    def generate_config(self, config_dir: Path, config: ParallelismConfig, model_name: str, backend: str, filename: Optional[str] = None, indent: int = 0, dim: bool = False) -> Path:
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
                           f"Available: {self.MODEL_FLAVORS[model_name]}", indent=indent, dim=dim)

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

        if self.verbose:
            log_message(LogLevel.INFO, f"Created config file: {config_file} for config '{config.name}' (model: {model_name})", indent=indent, dim=dim)
        return config_file
    
    def extract_metrics(self, log_file: Path, indent: int = 0, dim: bool = False) -> TrainingMetrics:
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
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}: {e}", indent=indent, dim=dim)
        
        if not metrics.loss or not metrics.grad_norm:
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}", indent=indent, dim=dim)
        
        return metrics
    
    def compare_metrics(self, baseline_metrics: TrainingMetrics, test_metrics: TrainingMetrics, 
                       config_name: str, indent: int = 0, dim: bool = False) -> bool:
        """Compare metrics between baseline and test configuration."""
        if not baseline_metrics.loss or not test_metrics.loss:
            log_message(LogLevel.TEST_FAIL, f"{config_name} - Unable to extract metrics", indent=indent, dim=dim)
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
        loss_max_diff = torch.max(torch.abs(baseline_loss - test_loss)).item() if baseline_loss.numel() > 0 and test_loss.numel() > 0 else 0.0
        grad_norm_diff = torch.max(torch.abs(baseline_grad_norm - test_grad_norm)).item() if baseline_grad_norm.numel() > 0 and test_grad_norm.numel() > 0 else 0.0
        
        # Calculate min absolute differences for logging
        loss_min_diff = torch.min(torch.abs(baseline_loss - test_loss)).item() if baseline_loss.numel() > 0 and test_loss.numel() > 0 else 0.0
        grad_norm_min_diff = torch.min(torch.abs(baseline_grad_norm - test_grad_norm)).item() if baseline_grad_norm.numel() > 0 and test_grad_norm.numel() > 0 else 0.0

        if loss_pass and grad_pass:
            log_message(LogLevel.TEST_PASS, 
                       f"{config_name} - Max loss diff: {loss_max_diff:.2e}, "
                       f"Min loss diff: {loss_min_diff:.2e}, "
                       f"Max grad norm diff: {grad_norm_diff:.2e}, "
                       f"Min grad norm diff: {grad_norm_min_diff:.2e}", indent=indent, dim=dim)
            return True
        else:
            log_message(LogLevel.TEST_FAIL,
                       f"{config_name} - Max loss diff: {loss_max_diff:.2e}, "
                       f"Min loss diff: {loss_min_diff:.2e}, "
                       f"Max grad norm diff: {grad_norm_diff:.2e}, "
                       f"Min grad norm diff: {grad_norm_min_diff:.2e}", indent=indent, dim=dim)
            return False
    
    def generate_diff(self, baseline_log: Path, test_log: Path, diff_file: Path, indent: int = 0, dim: bool = False) -> None:
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
            log_message(LogLevel.WARNING, f"Could not generate diff: {e}", indent=indent, dim=dim)
    
    def run_training(self, config_file: Path, log_file: Path, config_name: str, model_name: str, indent: int = 0, dim: bool = False) -> Optional[subprocess.CalledProcessError]:
        """Run training with given configuration."""
        log_message(LogLevel.INFO, f"Running training: {config_name} with model {model_name}", indent=indent, dim=dim)
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.ngpu}",
            "--rdzv_backend", "c10d",
            "--rdzv_endpoint=localhost:0",
            "--local-ranks-filter", str(self.ngpu - 1),
            "--role", "rank",
            "--tee", "3",
            "-m", "torchtitan.train",
            "--training.seed", str(self.seed),
            "--training.deterministic",
            "--job.config_file", str(config_file)
        ]
        env = os.environ.copy()
        env["SEED"] = str(self.seed)
        env["LOG_RANK"] = str(self.ngpu - 1)

        log_message(LogLevel.COMMAND, f"{' '.join(cmd)}", indent=indent, dim=dim)

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
                log_message(LogLevel.SUCCESS, f"Training completed: {config_name}", indent=indent, dim=dim)
            return None
            
        except subprocess.CalledProcessError as e:
            log_message(LogLevel.ERROR, f"Training failed: {config_name}", indent=indent, dim=dim)
            
            # Write the failed output to the log file
            with open(log_file, 'w') as f:
                if e.stdout:
                    f.write(e.stdout)

            # Print the tail of the error log to the console for quick debugging
            if e.stdout:
                console.print("[bold red]--- Error Log Tail ---[/bold red]")
                error_lines = e.stdout.strip().split('\n')
                for line in error_lines[-15:]:
                    console.print(f"[red]{line}[/red]")
                console.print("[bold red]--- End Error Log Tail ---[/bold red]")

            e.add_note(f"\n--- Full output from failed process ---\n{e.stdout or '<no output captured>'}")
            return e
    
    def _compare_one_parallelism_config(
        self,
        config: "ParallelismConfig",
        hf_model_name: str,
        tt_model_name: str,
        hf_baseline_metrics: "TrainingMetrics",
        tt_baseline_metrics: "TrainingMetrics",
        baseline_log_hf: Path,
        baseline_log_tt: Path,
        indent: int = 0,
    ) -> bool:
        """Compares a single parallelism configuration against the baseline."""
        # New flow: launch all training, then all diff, then all extract/compare metrics

        # --- 1. Setup directories and config files ---
        test_dir_name = f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface"
        test_dir = self.results_dir / test_dir_name
        test_dir.mkdir(exist_ok=True)

        config_filename_hf = f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.toml"
        config_file_hf = self.generate_config(
            config_dir=test_dir,
            config=config,
            model_name=hf_model_name,
            backend="huggingface",
            filename=config_filename_hf,
            indent=indent,
        )
        log_path_hf = test_dir / f"{config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.log"

        config_filename_tt = test_dir / f"{config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.toml"
        config_file_tt = self.generate_config(
            config_dir=test_dir,
            config=config,
            model_name=tt_model_name,
            backend="torchtitan",
            filename=config_filename_tt,
            indent=indent + 5,
            dim=True,
        )
        log_path_tt = test_dir / f"{config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.log"

        # --- 2. Launch all training (HF and TT) ---
        hf_run_error = self.run_training(
            config_file=config_file_hf,
            log_file=log_path_hf,
            config_name=config.name,
            model_name=hf_model_name,
            indent=indent,
        )
        tt_run_error = self.run_training(
            config_file=config_file_tt,
            log_file=log_path_tt,
            config_name=config.name,
            model_name=tt_model_name,
            indent=indent + 5,
            dim=True,
        )

        # If either training failed, log and skip further steps for this config
        if hf_run_error:
            log_message(
                LogLevel.TEST_FAIL,
                f"{config.name} (huggingface) - Training script failed.",
                indent=indent + 5,
                dim=True,
            )
            return False

        if tt_run_error:
            log_message(
                LogLevel.TEST_FAIL,
                f"{config.name} (torchtitan) - Training script failed.",
                indent=indent + 5,
                dim=True,
            )
            return False

        # --- 3. Generate all diffs ---
        list_of_diffs = {
            "HF baseline vs HF nd-parallel": (baseline_log_hf, log_path_hf, test_dir / "diff_hf_baseline_vs_hf_nd_parallelism.log"),
            "TT nd-parallel vs HF nd-parallel": (log_path_tt, log_path_hf, test_dir / "diff_tt_nd_parallelism_vs_hf_nd_parallelism.log"),
            "TT baseline vs HF nd-parallel": (baseline_log_tt, log_path_hf, test_dir / "diff_tt_baseline_vs_hf_nd_parallelism.log"),
            "TT baseline vs TT nd-parallel": (baseline_log_tt, log_path_tt, test_dir / "diff_tt_baseline_vs_tt_nd_parallelism.log"),
        }
        for src, dst, output in list_of_diffs.values():
            self.generate_diff(src, dst, output, indent=indent + 5, dim=True)

        # --- 4. Extract all metrics ---
        hf_metrics = self.extract_metrics(log_path_hf, indent=indent)
        tt_metrics = self.extract_metrics(log_path_tt, indent=indent + 5, dim=True)

        # --- 5. Compare metrics and determine pass/fail ---
        test_passed = True

        for diff_name, (src, dst, output) in list_of_diffs.items():
            if "TT nd-parallel vs HF nd-parallel" == diff_name:
                metrics_passed = self.compare_metrics(
                    tt_metrics,
                    hf_metrics,
                    diff_name,
                    indent=indent + 5,
                    dim=True,
                )
            elif "TT baseline vs TT nd-parallel" == diff_name:
                metrics_passed = self.compare_metrics(
                    tt_baseline_metrics,
                    tt_metrics,
                    diff_name,
                    indent=indent + 5,
                    dim=True,
                )
            elif "TT baseline vs HF nd-parallel" == diff_name:
                metrics_passed = self.compare_metrics(
                    tt_baseline_metrics,
                    hf_metrics,
                    diff_name,
                    indent=indent + 5,
                    dim=True,
                )
            else:  # HF baseline vs HF nd-parallel == diff_name
                metrics_passed = self.compare_metrics(
                    hf_baseline_metrics,
                    hf_metrics,
                    diff_name,
                    indent=indent + 5,
                    dim=True,
                )

            if not metrics_passed:
                test_passed = False

            log_message(
                LogLevel.INFO,
                f"Diff between {diff_name} saved to: {output}",
                indent=indent + 10,
                dim=True,
            )

        return test_passed

    def run(self) -> int:
        """Main execution function. Runs all test suites for all models."""
        parser = argparse.ArgumentParser(
            description="Test different parallelism configurations against a baseline FSDP model.",
        )
        parser.add_argument("-m", "--model-filter", default="",
                          help="Filter models by name pattern (e.g., 'llama3')")
        parser.add_argument("-t", "--test-filter", default="",
                          help="Filter parallelism configurations by name pattern (e.g., 'fsdp1_cp1_tp2_pp2')")
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
        self.test_filter = args.test_filter
        self.flavor = args.flavor
        self.verbose = args.verbose
        self.loss_atol = args.loss_atol
        self.loss_rtol = args.loss_rtol
        self.grad_norm_atol = args.grad_norm_atol
        self.grad_norm_rtol = args.grad_norm_rtol

        console.print(
            Panel(
                (
                    f"[bold]GPUs:[/bold] {self.ngpu}\n"
                    f"[bold]Steps:[/bold] {self.steps}\n"
                    f"[bold]Seed:[/bold] {self.seed}\n"
                    f"[bold]Model filter:[/bold] {self.model_filter or 'all'}\n"
                    f"[bold]Test filter:[/bold] {self.test_filter or 'all'}\n"
                    f"[bold]Model flavor:[/bold] {self.flavor}"
                ),
                title="[bold cyan]Distributed Parallelism Comparison[/bold cyan]",
                expand=False,
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()

        self.base_results_dir.mkdir(exist_ok=True)

        # TODO(3outeille): make it more generic later
        if self.model_filter == "llama3":
            hf_model_name = "meta-llama/Llama-3.2-1B"
            tt_model_name = "llama3"
        elif self.model_filter == "deepseek_v3":
            hf_model_name = "deepseek-ai/DeepSeek-V3"
            tt_model_name = "deepseek_v3"
        else:
            raise ValueError(f"Model filter {self.model_filter} not supported")
            
        self.generate_parallelism_configs(hf_model_name)
            
        model_owner, model_repo = hf_model_name.split("/", 1)
        nd_parallel_upper = self.nd_parallel.upper()
        self.results_dir = self.base_results_dir / model_owner / model_repo / nd_parallel_upper / self.flavor
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            log_message(LogLevel.INFO, f"Results directory: {self.results_dir}")

        console.print(
            Panel(
                "[bold cyan]Comparing baseline (FSDP) for huggingface & torchtitan[/bold cyan]",
                expand=False,
                border_style="blue",
                padding=(0, 2),
            )
        )

        baseline_config = next((c for c in self.parallelism_configs if c.name == "fsdp"), None)
        # --- 1. Generate configs ---
        baseline_config_filename_hf = f"baseline_{baseline_config.name}_{self.flavor}_{self.ngpu}gpu_huggingface.toml"
        baseline_config_file_hf = self.generate_config(
            config_dir=self.results_dir,
            config=baseline_config,
            model_name=hf_model_name,
            backend="huggingface",
            filename=baseline_config_filename_hf,
            indent=0
        )
        baseline_log_hf = self.results_dir / f"baseline_hf_{baseline_config.name}_{self.ngpu}gpu.log"

        baseline_config_filename_tt = f"baseline_{baseline_config.name}_{self.flavor}_{self.ngpu}gpu_torchtitan.toml"
        baseline_config_file_tt = self.generate_config(
            config_dir=self.results_dir,
            config=baseline_config,
            model_name=tt_model_name,
            backend="torchtitan", 
            filename=baseline_config_filename_tt,
            indent=0
        )
        baseline_log_tt = self.results_dir / f"baseline_tt_{baseline_config.name}_{self.ngpu}gpu.log"

        # --- 2. Launch all training ---
        hf_baseline_run_error = self.run_training(
            config_file=baseline_config_file_hf,
            log_file=baseline_log_hf,
            config_name=baseline_config.name,
            model_name=hf_model_name,
            indent=0
        )
        if hf_baseline_run_error:
            raise ValueError(f"Huggingface baseline (FSDP) training failed for {hf_model_name}") from hf_baseline_run_error

        tt_baseline_run_error = self.run_training(
            config_file=baseline_config_file_tt,
            log_file=baseline_log_tt,
            config_name=baseline_config.name,
            model_name=tt_model_name,
            indent=0
        )
        if tt_baseline_run_error:
            raise ValueError(f"TorchTitan baseline (FSDP) training failed for {tt_model_name}") from tt_baseline_run_error

        # --- 3. Generate diff ---
        diff_file_tt_baseline_vs_hf_baseline = self.results_dir / "diff_tt_baseline_vs_hf_baseline.log"
        self.generate_diff(
            baseline_log_tt,
            baseline_log_hf,
            diff_file_tt_baseline_vs_hf_baseline,
            indent=0
        )
        log_message(
            LogLevel.INFO,
            f"Diff between baseline TT and baseline HF saved to: {diff_file_tt_baseline_vs_hf_baseline}",
            indent=5,
            dim=True
        )

        # --- 4. Extract metrics ---
        hf_baseline_metrics = self.extract_metrics(baseline_log_hf, indent=0)
        if not hf_baseline_metrics.loss or not hf_baseline_metrics.grad_norm:
            raise ValueError(f"Could not extract huggingface baseline metrics for {hf_model_name}")

        tt_baseline_metrics = self.extract_metrics(baseline_log_tt, indent=0)
        if not tt_baseline_metrics.loss or not tt_baseline_metrics.grad_norm:
            raise ValueError(f"Could not extract TorchTitan baseline metrics for {tt_model_name}")

        # --- 5. Compare metrics ---
        if not self.compare_metrics(
            tt_baseline_metrics,
            hf_baseline_metrics,
            "baseline (TT) vs baseline (HF)",
            indent=5
        ):
            raise ValueError(f"Baseline (TT) vs baseline (HF) metrics comparison failed for {tt_model_name}")

        console.print()
        console.print(
            Panel(
                "[bold cyan]Comparing ND Parallelism Configurations[/bold cyan]",
                expand=False,
                border_style="blue",
                padding=(0, 2),
            )
        )
        passed_tests = 1 # +1 for the baseline (FSDP)
        failed_tests = 0
        test_configs = [c for c in self.parallelism_configs if c.name != "fsdp"]
        if self.test_filter:
            filtered_configs = [c for c in test_configs if self.test_filter in c.name]
            if not filtered_configs:
                log_message(LogLevel.WARNING, f"Test filter '{self.test_filter}' did not match any test configurations.")
            test_configs = filtered_configs
        total_tests = len(test_configs) + 1 # +1 for the baseline (FSDP)
        results = []

        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Comparing configurations...", total=total_tests
            )
            for i, config in enumerate(test_configs):
                if i > 0:
                    console.rule(style="dim")

                progress.update(
                    task, description=f"[cyan]Testing [bold]{config.name}[/bold]"
                )
                passed = self._compare_one_parallelism_config(
                    config,
                    hf_model_name,
                    tt_model_name,
                    hf_baseline_metrics,
                    tt_baseline_metrics,
                    baseline_log_hf,
                    baseline_log_tt,
                    indent=1,
                )
                results.append((config.name, passed))
                if passed:
                    passed_tests += 1
                else:
                    failed_tests += 1
                progress.advance(task)
        console.print()

        console.print(
            Panel(
                "[bold cyan]Final Summary[/bold cyan]",
                expand=False,
                border_style="blue",
                padding=(0, 2),
            )
        )

        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Configuration", style="cyan")
        summary_table.add_column("Status", justify="center")

        for name, passed in results:
            status = (
                "[bold green]✅ PASS[/bold green]"
                if passed
                else "[bold red]❌ FAIL[/bold red]"
            )
            summary_table.add_row(name, status)

        console.print(summary_table)
        console.print()

        overall_summary = Table(title="Overall Test Summary")
        overall_summary.add_column("Metric", style="cyan")
        overall_summary.add_column("Value", justify="right")
        overall_summary.add_row("Total Configurations Tested", str(total_tests))
        overall_summary.add_row("[green]Passed[/green]", str(passed_tests))
        overall_summary.add_row("[red]Failed[/red]", str(failed_tests))
        console.print(overall_summary)

        if passed_tests == total_tests:
            log_message(LogLevel.SUCCESS, "All model tests passed! 🎉")
            return 0
        else:
            log_message(LogLevel.TEST_FAIL, f"{failed_tests} configuration(s) had test failures")
            log_message(
                LogLevel.INFO, f"Check the diff files in {self.results_dir} for details"
            )
            return 1


def main():
    """Entry point for the script."""
    runner = CompareDistributedRun()
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())
