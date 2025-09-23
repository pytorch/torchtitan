#!/usr/bin/env python3
"""
compare_distributed_run.py - Test different parallelism configurations against baseline
Based on TorchTitan convergence guidelines

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
import tempfile
import json
from dataclasses import dataclass
from enum import Enum
import logging

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
    loss: Optional[float] = None
    grad_norm: Optional[float] = None

class CompareDistributedRun:
    """Main class for running distributed parallelism comparison tests."""
    
    # Default values
    DEFAULT_THRESHOLD_LOSS = 1e-4
    DEFAULT_THRESHOLD_GRAD_NORM = 1e-3
    DEFAULT_STEPS = 10
    DEFAULT_SEED = 42
    DEFAULT_FLAVOR = "debugmodel"
    
    # HF Model lists - extendable for different model families
    HF_MODEL_LISTS = {
        "llama": "meta-llama/Llama-3.2-1B",
        "deepseek": "deepseek-ai/DeepSeek-V3",
    }
    
    # Available flavors per model type
    MODEL_FLAVORS = {
        "llama": ["debugmodel", "medium", "full"],
        "deepseek": ["debugmodel"],
    }

    # Available ND parallelisms <-> number of GPUs
    ND_PARALLEL_TO_NB_GPUS = {
        "1d": 2,
        "2d": 4,
        "3d": 8,
        "4d": 16,
    }
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.torchtitan_root = self.script_dir.parent.parent
        self.results_dir = self.script_dir / "comparison_results"
        self.config_dir = self.script_dir / "generated_configs"
        
        # Configuration parameters
        self.loss_threshold = self.DEFAULT_THRESHOLD_LOSS
        self.grad_norm_threshold = self.DEFAULT_THRESHOLD_GRAD_NORM
        self.nd_parallel_to_nb_gpus = self.ND_PARALLEL_TO_NB_GPUS
        self.steps = self.DEFAULT_STEPS
        self.seed = self.DEFAULT_SEED
        self.model_filter = ""
        self.flavor = self.DEFAULT_FLAVOR
        self.verbose = False
        self.parallelism_configs: List[ParallelismConfig] = []

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

        possible_tp = _get_factors(ngpu)
        possible_pp = _get_factors(ngpu)
        possible_ep = _get_factors(ngpu)
        #TODO(3outeille): is CP borrowing degree from DP ?
        #TODO(3outeille): is EP borrowing degree from DP ? 

        # Is that correct ?
        for tp in possible_tp:
            for pp in possible_pp:
                for ep in possible_ep:
                    if tp * pp * ep > ngpu:
                        continue

                    if ngpu % (tp * pp * ep) == 0:
                        dp = ngpu // (tp * pp * ep)
                        if dp > 0 and (tp > 1 or pp > 1 or ep > 1 or dp > 1):
                            # DDP style
                            if dp > 1:
                                configs.append(
                                    ParallelismConfig(
                                        name=f"tp{tp}_pp{pp}_ep{ep}_ddp{dp}",
                                        dp_replicate=dp,
                                        dp_shard=1,
                                        tp=tp,
                                        pp=pp,
                                        pp_schedule="Interleaved1F1B",
                                        cp=1,
                                        ep=ep,
                                        eptp=1
                                    )
                                )
                            # FSDP with other parallelisms
                            if tp > 1 or pp > 1 or ep > 1:
                                configs.append(
                                    ParallelismConfig(
                                        name=f"tp{tp}_pp{pp}_ep{ep}_fsdp",
                                        dp_replicate=1,
                                        dp_shard=-1,
                                        tp=tp,
                                        pp=pp,
                                        pp_schedule="Interleaved1F1B",
                                        cp=1,
                                        ep=ep,
                                        eptp=1
                                    )
                                )

        # HSDP requires a DP degree that can be split
        for dp in _get_factors(ngpu):
            if dp > 1:
                dp_factors = _get_factors(dp)
                for replicate in dp_factors:
                    if replicate > 1:
                        shard = dp // replicate
                        if shard > 1:
                            configs.append(
                                ParallelismConfig(
                                    name=f"hsdp_r{replicate}_s{shard}",
                                    dp_replicate=replicate,
                                    dp_shard=shard,
                                    tp=1,
                                    pp=1,
                                    pp_schedule="Interleaved1F1B",
                                    cp=1,
                                    ep=1,
                                    eptp=1
                                )
                            )
        
        # Remove duplicates and assign to instance
        unique_configs = []
        seen_configs = set()
        for config in configs:
            # Create a tuple of the config values to check for duplicates
            config_tuple = (config.dp_replicate, config.dp_shard, config.tp, config.pp, config.ep)
            if config_tuple not in seen_configs:
                unique_configs.append(config)
                seen_configs.add(config_tuple)

        self.parallelism_configs = unique_configs
        
        log_message(LogLevel.INFO, f"Generated {len(self.parallelism_configs)} parallelism configurations for {ngpu} GPUs.")
        if self.verbose:
            for config in self.parallelism_configs:
                log_message(LogLevel.INFO, f"  - {config.name}: dp_replicate={config.dp_replicate}, dp_shard={config.dp_shard}, tp={config.tp}, pp={config.pp}, ep={config.ep}")
    def generate_config(self, config: ParallelismConfig, model_name: str, model_type: str) -> Path:
        """Generate configuration file for a parallelism setup."""
        config_file = self.config_dir / f"{config.name}_{model_type}_{self.flavor}_{self.nd_parallel_to_nb_gpus[self.nd_parallel]}gpu.toml"
        
        #TODO(3outeille): create template instead
        if model_type == "llama":
            base_config = self.script_dir / "configs" / "debug_1_gpu_tt.toml"
        else:
            base_config = self.script_dir / "configs" / "debug_1_gpu_hf.toml"
        
        shutil.copy2(base_config, config_file)

        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update model name if it's HF backend
        if model_type != "llama":
            content = re.sub(r'name = ".*"', f'name = "{model_name}"', content)
        
        # Update model flavor
        content = re.sub(r'flavor = ".*"', f'flavor = "{self.flavor}"', content)
        
        # Validate flavor for model type
        if model_type in self.MODEL_FLAVORS:
            if self.flavor not in self.MODEL_FLAVORS[model_type]:
                log_message(LogLevel.WARNING, 
                           f"Flavor '{self.flavor}' not available for {model_type}. "
                           f"Available: {self.MODEL_FLAVORS[model_type]}")
        
        # Update training steps and seed
        content = re.sub(r'steps = .*', f'steps = {self.steps}', content)
        if 'seed = ' in content:
            content = re.sub(r'seed = .*', f'seed = {self.seed}', content)
        else:
            content = re.sub(r'(steps = .*)', f'\\1\nseed = {self.seed}', content)
        
        #TODO(3outeille): is this correct ?
        # Ensure deterministic training
        if 'deterministic = true' not in content:
            content = re.sub(r'(seed = .*)', '\\1\ndeterministic = true', content)
        
        # Update parallelism configuration
        content = re.sub(r'data_parallel_replicate_degree = .*', 
                        f'data_parallel_replicate_degree = {config.dp_replicate}', content)
        content = re.sub(r'data_parallel_shard_degree = .*', 
                        f'data_parallel_shard_degree = {config.dp_shard}', content)
        content = re.sub(r'tensor_parallel_degree = .*', 
                        f'tensor_parallel_degree = {config.tp}', content)
        content = re.sub(r'pipeline_parallel_degree = .*', 
                        f'pipeline_parallel_degree = {config.pp}', content)
        content = re.sub(r'pipeline_parallel_schedule = .*', 
                        f'pipeline_parallel_schedule = "{config.pp_schedule}"', content)
        content = re.sub(r'context_parallel_degree = .*', 
                        f'context_parallel_degree = {config.cp}', content)
        content = re.sub(r'expert_parallel_degree = .*', 
                        f'expert_parallel_degree = {config.ep}', content)
        
        content = re.sub(r'expert_tensor_parallel_degree = .*', 
                        f'expert_tensor_parallel_degree = {config.eptp}', content)

        # Write modified config
        with open(config_file, 'w') as f:
            f.write(content)
        
        log_message(LogLevel.INFO, f"Created config file: {config_file} for config '{config.name}' (model: {model_name}, type: {model_type})")
        return config_file
    
    def extract_metrics(self, log_file: Path) -> TrainingMetrics:
        """Extract metrics from log file."""
        metrics = TrainingMetrics()
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract final loss and grad_norm from the last step
            loss_matches = re.findall(r'loss:\s*([0-9]+\.?[0-9]*)', content)
            grad_norm_matches = re.findall(r'grad_norm:\s*([0-9]+\.?[0-9]*)', content)
            
            if loss_matches:
                metrics.loss = float(loss_matches[-1])
            if grad_norm_matches:
                metrics.grad_norm = float(grad_norm_matches[-1])
                
        except Exception as e:
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}: {e}")
        
        if metrics.loss is None or metrics.grad_norm is None:
            log_message(LogLevel.WARNING, f"Could not extract metrics from {log_file}")
        
        return metrics
    
    def compare_metrics(self, baseline_metrics: TrainingMetrics, test_metrics: TrainingMetrics, 
                       config_name: str) -> bool:
        """Compare metrics between baseline and test configuration."""
        if (baseline_metrics.loss is None or baseline_metrics.grad_norm is None or
            test_metrics.loss is None or test_metrics.grad_norm is None):
            log_message(LogLevel.TEST_FAIL, f"{config_name} - Unable to extract metrics")
            return False
        
        # Calculate absolute differences
        loss_diff = abs(baseline_metrics.loss - test_metrics.loss)
        grad_norm_diff = abs(baseline_metrics.grad_norm - test_metrics.grad_norm)
        
        # Check if differences are within thresholds
        loss_pass = loss_diff < self.loss_threshold
        grad_pass = grad_norm_diff < self.grad_norm_threshold
        
        if loss_pass and grad_pass:
            log_message(LogLevel.TEST_PASS, 
                       f"{config_name} - Loss diff: {loss_diff:.2e} (< {self.loss_threshold:.2e}), "
                       f"Grad norm diff: {grad_norm_diff:.2e} (< {self.grad_norm_threshold:.2e})")
            return True
        else:
            log_message(LogLevel.TEST_FAIL,
                       f"{config_name} - Loss diff: {loss_diff:.2e} (threshold: {self.loss_threshold:.2e}), "
                       f"Grad norm diff: {grad_norm_diff:.2e} (threshold: {self.grad_norm_threshold:.2e})")
            return False
    
    def generate_diff(self, baseline_log: Path, log_path: Path, diff_file: Path) -> None:
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
            test_filtered = _filter_log(log_path)
            
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
    
    def run_training(self, config_file: Path, log_file: Path, config_name: str, model_name: str) -> bool:
        """Run training with given configuration."""
        log_message(LogLevel.INFO, f"Running training: {config_name} with model {model_name}")
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.nd_parallel_to_nb_gpus[self.nd_parallel]}",
            "--rdzv_backend", "c10d",
            "--rdzv_endpoint=localhost:0",
            "--local-ranks-filter", "0",
            "--role", "rank",
            "--tee", "3",
            "-m", "torchtitan.train",
            "--job.config_file", str(config_file)
        ]
        
        env = os.environ.copy()
        
        if self.verbose:
            log_message(LogLevel.INFO, f"Command: {' '.join(cmd)}")
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=self.torchtitan_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True
                )
            
            if self.verbose:
                log_message(LogLevel.SUCCESS, f"Training completed: {config_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            log_message(LogLevel.ERROR, f"Training failed: {config_name}")
            return False
    
    def run(self) -> int:
        """Main execution function. Runs all test suites for all models."""
        parser = argparse.ArgumentParser(
            description="Test different parallelism configurations against a baseline FSDP model.",
        )      
        parser.add_argument("-m", "--model-filter", default="",
                          help="Filter models by name pattern (e.g., 'llama')")
        parser.add_argument("-t", "--loss-threshold", type=float, default=self.DEFAULT_THRESHOLD_LOSS,
                          help=f"Loss difference threshold (default: {self.DEFAULT_THRESHOLD_LOSS})")
        parser.add_argument("-g", "--grad-threshold", type=float, default=self.DEFAULT_THRESHOLD_GRAD_NORM,
                          help=f"Grad norm difference threshold (default: {self.DEFAULT_THRESHOLD_GRAD_NORM})")
        parser.add_argument("-nd", "--nd_parallel", type=str, default="2d",
                          help=f"Parallelism to use (default: {self.ND_PARALLEL_TO_NB_GPUS.keys()})")
        parser.add_argument("-s", "--steps", type=int, default=self.DEFAULT_STEPS,
                          help=f"Training steps (default: {self.DEFAULT_STEPS})")
        parser.add_argument("--seed", type=int, default=self.DEFAULT_SEED,
                          help=f"Random seed (default: {self.DEFAULT_SEED})")
        parser.add_argument("--flavor", default=self.DEFAULT_FLAVOR,
                          help=f"Model flavor/size (default: {self.DEFAULT_FLAVOR}). "
                               f"Available: llama=[debugmodel, medium, full], deepseek=[debugmodel]")
        parser.add_argument("-v", "--verbose", action="store_true",
                          help="Verbose output")
        
        args = parser.parse_args()
        
        self.loss_threshold = args.loss_threshold
        self.grad_norm_threshold = args.grad_threshold
        self.nd_parallel = args.nd_parallel
        self.steps = args.steps
        self.seed = args.seed
        self.model_filter = args.model_filter
        self.flavor = args.flavor
        self.verbose = args.verbose
        
        log_message(LogLevel.INFO, "=== TorchTitan Distributed Parallelism Comparison ===")
        log_message(LogLevel.INFO, f"Loss threshold: {self.loss_threshold}")
        log_message(LogLevel.INFO, f"Grad norm threshold: {self.grad_norm_threshold}")
        log_message(LogLevel.INFO, f"GPUs: {self.nd_parallel_to_nb_gpus[self.nd_parallel]}")
        log_message(LogLevel.INFO, f"Steps: {self.steps}")
        log_message(LogLevel.INFO, f"Seed: {self.seed}")
        log_message(LogLevel.INFO, f"Model filter: {self.model_filter or 'all'}")
        log_message(LogLevel.INFO, f"Model flavor: {self.flavor}")
        print()
        
        self.results_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            log_message(LogLevel.INFO, f"Results directory: {self.results_dir}")
            log_message(LogLevel.INFO, f"Config directory: {self.config_dir}")

        self.generate_parallelism_configs()
        
        total_model_failures = 0

        for model_type, model_name in self.HF_MODEL_LISTS.items():
            # Apply model filter if specified
            if self.model_filter and self.model_filter not in model_type:
                continue

            log_message(LogLevel.INFO, f"Testing model: {model_type} ({model_name})")
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            configs_to_run = []

            for config in self.parallelism_configs:
                # Skip configurations that require more GPUs than available
                required_gpus = config.dp_replicate * config.tp * config.pp
                if config.dp_shard != -1:
                    required_gpus *= config.dp_shard

                if required_gpus > self.nd_parallel_to_nb_gpus[self.nd_parallel]:
                    log_message(LogLevel.WARNING, 
                               f"Skipping {config.name}: requires {required_gpus} GPUs but only {self.ngpu} available")
                    continue

                config_file = self.generate_config(config, model_name, model_type)
                configs_to_run.append((config, config_file))

            # # Test each parallelism configuration
            # for config, config_file in configs_to_run:
            #     log_path = self.results_dir / f"{config.name}_{model_type}_{self.flavor}_{self.ngpu}gpu.log"
            #     if not self.run_training(config_file, log_path, config.name, model_name):
            #         log_message(LogLevel.TEST_FAIL, f"{config.name} - Training failed")
            #         failed_tests += 1
            #         continue
            #     test_metrics = self.extract_metrics(log_path)
            #     if self.compare_metrics(baseline_metrics, test_metrics, config.name):
            #         passed_tests += 1
            #     else:
            #         failed_tests += 1
            #         diff_file = self.results_dir / f"diff_{config.name}_vs_baseline_{model_type}_{self.flavor}_{self.ngpu}gpu.log"
            #         self.generate_diff(baseline_log, log_path, diff_file)
            #         log_message(LogLevel.INFO, f"Diff saved to: {diff_file}")
            #     total_tests += 1

            # Print summary for this model
            print()
            log_message(LogLevel.INFO, f"=== TEST SUMMARY for {model_type} ===")
            log_message(LogLevel.INFO, f"Total tests: {total_tests}")
            log_message(LogLevel.SUCCESS, f"Passed: {passed_tests}")
            if failed_tests > 0:
                log_message(LogLevel.TEST_FAIL, f"Failed: {failed_tests}")
            else:
                log_message(LogLevel.INFO, f"Failed: {failed_tests}")
            print()

            if failed_tests > 0:
                total_model_failures += 1

        # Final summary
        print()
        log_message(LogLevel.INFO, "=== FINAL SUMMARY ===")
        if total_model_failures == 0:
            log_message(LogLevel.SUCCESS, "All model tests passed! 🎉")
            return 0
        else:
            log_message(LogLevel.TEST_FAIL, f"{total_model_failures} model(s) had test failures")
            log_message(LogLevel.INFO, f"Check the diff files in {self.results_dir} for details")
            return 1

def main():
    """Entry point for the script."""
    runner = CompareDistributedRun()
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())
