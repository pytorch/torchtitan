import toml
from argparse import ArgumentParser
from pathlib import Path
import re
import os
import subprocess
from enum import Enum
from jinja2 import Template
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# BASELINE = "fsdp2_tp1_cp1_pp1"
BASELINE = "fsdp1_tp1_cp1_pp1"

console = Console()

class LogLevel(Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TEST_PASS = "TEST_PASS"
    TEST_FAIL = "TEST_FAIL"

def log_message(level: LogLevel, message: str, indent: int = 0, dim: bool = False) -> None:
    """Log a message with appropriate color coding."""
    style_map = {
        LogLevel.INFO: "blue",
        LogLevel.SUCCESS: "green",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "bold red",
        LogLevel.TEST_PASS: "green",
        LogLevel.TEST_FAIL: "bold red",
    }

    prefix_map = {
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
         
    output = f"{indent_str}[{style}]{prefix}[/] {message}"

    if dim:
        console.print(f"[dim]{output}[/dim]")
    else:
        console.print(output)


def _create_slurm_script(
    config: dict,
    config_path: Path,
    script_path: Path,
    job_name: str,
    initial_load_path: str = None,
    repo_id: str = None,
):
    with open(config_path, "r") as file:
        config = toml.load(file)

    pp = config["parallelism"]["pipeline_parallel_degree"]
    dp = config["parallelism"]["data_parallel_shard_degree"]
    tp = config["parallelism"]["tensor_parallel_degree"]
    cp = config["parallelism"]["context_parallel_degree"]
    world_size = pp * dp * tp * cp

    nodes = max(1, world_size // 8)
    n_proc_per_node = min(8, world_size // nodes)

    print(f"world_size: {world_size}, nodes: {nodes}, n_proc_per_node: {n_proc_per_node}")

    # Read the SLURM script template from the file
    template_path = Path(__file__).parent / "configs/template.slurm"
    with open(template_path, "r") as f:
        slurm_script_template = f.read()
    base_bench_template = Template(slurm_script_template)

    context_bench = {
        "name": job_name,
        "nodes": nodes,
        "n_proc_per_node": n_proc_per_node,
        "root_path": script_path.parent,
        "config_path": config_path,
        "initial_load_path": initial_load_path,
        "repo_id": repo_id,
        "qos": "high" if nodes > 1 else "normal",  # Example logic for qos
    }

    with open(script_path, "w") as file:
        file.write(base_bench_template.render(context_bench))

    print(f"Slurm script created at {script_path}")


def create_configs(model_name: str, out_dir: str, flavor: str):
    """
    results/
        |_ meta-llama
            |_ Llama-3.2-1B
                |_ debugmodel/
                    |_ seed_checkpoint/
                        |_ config.toml
                        |_ seed.slurm
                        |_ step-0/
                           |_ ....
                    |_ fsdp2_tp1_cp1_pp1/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                    |_ fsdp2_tp2_cp1_pp1/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                        |_ diff_baseline_vs_nd_parallelism.log
                    |_ fsdp2_tp1_cp1_pp2/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                        |_ diff_baseline_vs_nd_parallelism.log
                    |_ fsdp2_tp1_cp2_pp1/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                        |_ diff_baseline_vs_nd_parallelism.log
                    |_ fsdp2_tp1_cp2_pp2/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                        |_ diff_baseline_vs_nd_parallelism.log`
                |_ full/
                ...
        |_ llama3 #torchtitan model
    """

    base_config = "configs/test_template.toml"
    with open(base_config, "r") as f:
        config = toml.load(f)

    config["model"]["name"] = model_name
    config["model"]["flavor"] = flavor

    # parallelism_configs = [
    #     BASELINE, # baseline
    #     "fsdp2_tp2_cp1_pp1",
    #     # "fsdp2_tp1_cp1_pp2",
    #     # "fsdp2_tp1_cp2_pp1",
    #     # "fsdp2_tp1_cp2_pp2",
    #     # "fsdp2_tp2_cp2_pp1",
    #     # "fsdp2_tp2_cp1_pp2",
    #     # "fsdp2_tp2_cp2_pp2",
    # ]

    # parallelism_configs = [
    #     BASELINE, # baseline
    #     # "fsdp1_tp2_cp1_pp1",
    #     # "fsdp1_tp1_cp1_pp2",
    #     # "fsdp1_tp1_cp2_pp1",
    #     # "fsdp1_tp1_cp2_pp2",
    #     # "fsdp1_tp2_cp2_pp1",
    #     # "fsdp1_tp2_cp1_pp2",
    #     # "fsdp1_tp2_cp2_pp2",
    # ]

    parallelism_configs = [
        BASELINE, # baseline
        "fsdp1_tp2_cp1_pp1",
    ]

    out_path = Path(out_dir) / model_name / flavor
    out_path.mkdir(parents=True, exist_ok=True)

    # Create seed checkpoint
    seed_config = toml.loads(toml.dumps(config))
    seed_config["parallelism"]["data_parallel_shard_degree"] = 1
    seed_config["parallelism"]["tensor_parallel_degree"] = 1
    seed_config["parallelism"]["pipeline_parallel_degree"] = 1
    seed_config["parallelism"]["context_parallel_degree"] = 1
    seed_checkpoint_dir = out_path / "seed_checkpoint"
    seed_checkpoint_dir.mkdir(exist_ok=True)
    seed_config["job"]["dump_folder"] = str(seed_checkpoint_dir)
    seed_config_path = seed_checkpoint_dir / "config.toml"
    with open(seed_config_path, "w") as f:
        toml.dump(seed_config, f)
    print(f"Created {seed_config_path}")
    _create_slurm_script(
        seed_config,
        seed_config_path,
        seed_checkpoint_dir / "seed.slurm",
        "seed_checkpoint",
        repo_id=model_name,
    )

    # Create parallelism configs
    for pc in parallelism_configs:
            
        iter_config = toml.loads(toml.dumps(config))

        m = re.match(r"fsdp(\d+)_tp(\d+)_cp(\d+)_pp(\d+)", pc)
        if not m:
            print(f"Skipping invalid config string: {pc}")
            continue

        fsdp, tp, cp, pp = map(int, m.groups())

        pc_dir = out_path / pc
        pc_dir.mkdir(exist_ok=True)

        iter_config["parallelism"]["data_parallel_shard_degree"] = fsdp
        iter_config["parallelism"]["tensor_parallel_degree"] = tp
        iter_config["parallelism"]["context_parallel_degree"] = cp
        iter_config["parallelism"]["pipeline_parallel_degree"] = pp
        iter_config["parallelism"]["pipeline_parallel_schedule"] = "GPipe"
        iter_config["job"]["dump_folder"] = str(pc_dir)
        
        # if pc == "fsdp1_tp1_cp1_pp2" or pc == BASELINE:
        #     iter_config["training"]["global_batch_size"] = 1
        #     iter_config["training"]["local_batch_size"] = 1

        if pc == BASELINE or pc == "fsdp2_tp1_cp1_pp2":
            iter_config["training"]["local_batch_size"] = 2

        config_path = pc_dir / "config.toml"
        with open(config_path, "w") as f:
            toml.dump(iter_config, f)
        print(f"Created {config_path}")
        _create_slurm_script(
            iter_config,
            config_path,
            pc_dir / "nd_parallelism.slurm",
            pc,
            initial_load_path=str(seed_checkpoint_dir / "checkpoint/step-0"),
            repo_id=model_name,
        )

class Status(Enum):
    # INIT -> PENDING -> [RUNNING | FAIL] -> COMPLETED
    INIT = "init"  # Job is created
    PENDING = "pending"  # Job is waiting for ressources
    RUNNING = "running"  # Job is running
    FAIL = "fail"  # Job failed
    COMPLETED = "completed"  # Job is completed

class Job:
    def __init__(self, root_path: str, qos: str, inp_dir: str = None) -> None:
        self.root_path = root_path
        self.name = os.path.basename(root_path)
        
        self.config = os.path.join(root_path, "config.toml")
        seed_slurm = os.path.join(root_path, "seed.slurm")
        if os.path.exists(seed_slurm):
            self.slurm_script = seed_slurm
        else:
            self.slurm_script = os.path.join(root_path, "nd_parallelism.slurm")

        self.qos = qos

        # Check if the status.txt file exists
        status_file_path = os.path.join(self.root_path, "status.txt")
        if not os.path.exists(status_file_path):
            # Create the status.txt file with INIT status
            with open(status_file_path, "w") as f:
                f.write(Status.INIT.value)
        self.status = self.get_status()

    def get_status(self) -> Status:
        """
        Read the status of the job from `status.txt` and return it
        """
        is_existing = lambda value_to_check: any(
            value.value == value_to_check for value in Status.__members__.values()
        )

        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, "r") as f:
            status = f.read().strip()
            if not is_existing(status):
                raise ValueError(f"Invalid status: {status}")
            return Status(status)

    def set_status(self, status: Status) -> Status:
        """
        Update the status of the job in `status.txt` and return the new status
        """
        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, "w") as f:
            f.write(status.value)
            return status

class Scheduler:
    def __init__(self, inp_dir: str, qos: str) -> None:
        # Find all leaf directories, and the top-level directory if it contains a config.
        jobs_directory_paths = []
        for root, dirs, files in os.walk(inp_dir):
            is_job_dir = any(f.endswith(".toml") for f in files)
            if is_job_dir:
                if not dirs: # leaf node
                    jobs_directory_paths.append(os.path.abspath(root))
                # also capture baseline job in root
                elif root == inp_dir:
                    jobs_directory_paths.append(os.path.abspath(root))

        self.job_lists = [Job(job_path, qos, inp_dir) for job_path in jobs_directory_paths]

    def keep_only_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status == status]

    def filter_out_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status != status]


def submit_jobs(inp_dir, qos, only: str = None):
    scheduler = Scheduler(inp_dir, qos)

    env_vars = os.environ.copy()
    total_jobs = len(scheduler.job_lists)

    if only:
        try:
            status_to_filter = Status(only)
            scheduler.job_lists = scheduler.keep_only_jobs(status_to_filter)
        except ValueError:
            print(f"Invalid status for --only: {only}")
            return

    if only is not None:
        filtered_jobs = len(scheduler.job_lists)
        if filtered_jobs == 0:
            print(f"No '{only}' jobs to resubmit")
            return
        print(
            f"Only {filtered_jobs}/{total_jobs} jobs with status '{only}' will be resubmitted"
        )

    scheduler.job_lists = scheduler.filter_out_jobs(Status.COMPLETED)

    for job in scheduler.job_lists:
        subprocess.run(["sbatch", job.slurm_script], env=env_vars)
        job.set_status(Status.PENDING)


def check_status(inp_dir: str):
    """
    Display a table showing the count of jobs in each status.
    Reads status.txt from all job directories found in inp_dir.
    """
    # Find all directories with status.txt files
    jobs_directory_paths = []
    for root, dirs, files in os.walk(inp_dir):
        if "status.txt" in files:
            jobs_directory_paths.append(os.path.abspath(root))
    
    if not jobs_directory_paths:
        print(f"No jobs found in {inp_dir}")
        return
    
    # Count jobs by status
    status_counts = {status: 0 for status in Status}
    for job_path in jobs_directory_paths:
        job = Job(job_path, qos="N/A")
        status_counts[job.status] += 1
    
    total = len(jobs_directory_paths)
    
    # Print table
    print("\nJob Status Summary")
    print("=" * 30)
    print(f"{'Status':<12} | {'Count':>5}")
    print("-" * 30)
    print(f"{'Init':<12} | {status_counts[Status.INIT]:>5}")
    print(f"{'Pending':<12} | {status_counts[Status.PENDING]:>5}")
    print(f"{'Running':<12} | {status_counts[Status.RUNNING]:>5}")
    print(f"{'Fail':<12} | {status_counts[Status.FAIL]:>5}")
    print(f"{'Completed':<12} | {status_counts[Status.COMPLETED]:>5}")
    print("-" * 30)
    print(f"{'Total':<12} | {total:>5}")
    print("=" * 30)


def report(inp_dir: str, only: str = None):
    """
    Generate diff reports between baseline (fsdp2_tp1_cp1_pp1) and all other parallelism configs.
    Creates diff_baseline_vs_nd_parallelism.log in each non-baseline config directory.
    Automatically discovers all model/flavor combinations under inp_dir.
    """
    # Add imports
    import torch
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class TrainingMetrics:
        """Training metrics extracted from logs."""
        steps: List[int] = field(default_factory=list)
        loss: List[float] = field(default_factory=list)
        grad_norm: List[float] = field(default_factory=list)
    
    # Default tolerance values (matching compare_distributed_run.py)
    DEFAULT_LOSS_ATOL = 5e-2
    DEFAULT_LOSS_RTOL = 1e-5
    DEFAULT_GRAD_NORM_ATOL = 7e-1
    DEFAULT_GRAD_NORM_RTOL = 1e-5
    
    def _extract_metrics(log_file: Path) -> TrainingMetrics:
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
            log_message(LogLevel.WARNING, f"Could not extract metrics: {e}", indent=3, dim=True)
        
        return metrics
    
    def _compare_metrics(baseline_metrics: TrainingMetrics, test_metrics: TrainingMetrics, 
                        config_name: str) -> tuple[bool, str]:
        """Compare metrics between baseline and test configuration.
        
        Returns:
            tuple[bool, str]: (passed, summary_message)
        """
        if not baseline_metrics.loss or not test_metrics.loss:
            return False, f"Unable to extract metrics"
        
        # Convert to tensors
        baseline_loss = torch.tensor(baseline_metrics.loss)
        test_loss = torch.tensor(test_metrics.loss)
        baseline_grad_norm = torch.tensor(baseline_metrics.grad_norm)
        test_grad_norm = torch.tensor(test_metrics.grad_norm)
        
        # Check if tensors are close
        loss_pass = torch.allclose(baseline_loss, test_loss, atol=DEFAULT_LOSS_ATOL, rtol=DEFAULT_LOSS_RTOL)
        grad_pass = torch.allclose(baseline_grad_norm, test_grad_norm, atol=DEFAULT_GRAD_NORM_ATOL, rtol=DEFAULT_GRAD_NORM_RTOL)

        # Calculate max absolute differences for logging
        loss_max_diff = torch.max(torch.abs(baseline_loss - test_loss)).item() if baseline_loss.numel() > 0 and test_loss.numel() > 0 else 0.0
        grad_norm_diff = torch.max(torch.abs(baseline_grad_norm - test_grad_norm)).item() if baseline_grad_norm.numel() > 0 and test_grad_norm.numel() > 0 else 0.0
        
        # Calculate min absolute differences for logging
        loss_min_diff = torch.min(torch.abs(baseline_loss - test_loss)).item() if baseline_loss.numel() > 0 and test_loss.numel() > 0 else 0.0
        grad_norm_min_diff = torch.min(torch.abs(baseline_grad_norm - test_grad_norm)).item() if baseline_grad_norm.numel() > 0 and test_grad_norm.numel() > 0 else 0.0

        summary = (f"Max loss diff: {loss_max_diff:.2e}, "
                  f"Min loss diff: {loss_min_diff:.2e}, "
                  f"Max grad norm diff: {grad_norm_diff:.2e}, "
                  f"Min grad norm diff: {grad_norm_min_diff:.2e}")
        
        return (loss_pass and grad_pass), summary

    def _filter_log(log_file: Path) -> Path:
        """Filter log file to normalize volatile information (timestamps, PIDs, ports)."""
        filtered_file = log_file.with_suffix(log_file.suffix + '.filtered')
        
        with open(log_file, 'r') as infile, open(filtered_file, 'w') as outfile:
            for line in infile:
                # Apply filtering patterns to remove volatile information
                line = re.sub(r'([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?', 
                            'TIMESTAMP', line)
                line = re.sub(r'torchrun.*--master_port[= ]([0-9]+)', 
                            'torchrun ... --master_port=XXXX', line)
                line = re.sub(r'PID [0-9]+', 'PID XXXX', line)
                line = re.sub(r'localhost:[0-9]+', 'localhost:XXXX', line)
                outfile.write(line)
        
        return filtered_file

    def _generate_diff(baseline_log: Path, test_log: Path, diff_file: Path) -> tuple[bool, str]:
        """Generate diff between baseline and test logs using git diff.
        
        Returns:
            tuple[bool, str]: (success, diff_output or error_message)
        """
        # Filter logs to remove timestamps and volatile information
        baseline_filtered = _filter_log(baseline_log)
        test_filtered = _filter_log(test_log)
        
        try:
            # Generate colored diff using git diff
            cmd = ["git", "diff", "--no-index", "--color=always", "--word-diff=color",
                str(baseline_filtered), str(test_filtered)]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # git diff returns exit code 1 when files differ (which is expected), not an error
            if result.returncode not in [0, 1]:
                error_msg = f"git diff failed with code {result.returncode}\n{result.stderr}"
                return False, error_msg
            
            # Write diff to file
            with open(diff_file, 'w') as f:
                f.write(result.stdout)
            
            return True, result.stdout
            
        finally:
            # Clean up filtered files
            if baseline_filtered.exists():
                baseline_filtered.unlink()
            if test_filtered.exists():
                test_filtered.unlink()

    def _process_flavor_dir(flavor_dir: Path) -> tuple[int, int]:
        """Process a single model/flavor directory.
        
        Returns:
            tuple[int, int]: (passed_count, failed_count)
        """
        # Find baseline directory
        baseline_dir = flavor_dir / BASELINE
        if not baseline_dir.exists():
            log_message(LogLevel.WARNING, f"No baseline directory found in {flavor_dir.relative_to(inp_path)}, skipping", indent=1)
            return 0, 0
        
        # Find baseline .out file
        baseline_out_files = list(baseline_dir.glob("*.out"))
        if not baseline_out_files:
            log_message(LogLevel.WARNING, f"No .out file found in baseline {baseline_dir.relative_to(inp_path)}, skipping", indent=1)
            return 0, 0
        baseline_out = baseline_out_files[0]
        
        # Extract baseline metrics
        log_message(LogLevel.INFO, f"Extracting baseline metrics from {baseline_out.name}...", indent=1)
        baseline_metrics = _extract_metrics(baseline_out)
        if not baseline_metrics.loss or not baseline_metrics.grad_norm:
            log_message(LogLevel.WARNING, "Could not extract baseline metrics, skipping comparisons", indent=1)
            return 0, 0
        
        # Find all parallelism config directories (excluding seed_checkpoint and baseline)
        config_dirs = []
        for item in flavor_dir.iterdir():
            if item.is_dir() and item.name not in {BASELINE, "seed_checkpoint"}:
                config_dirs.append(item)
        
        if not config_dirs:
            log_message(LogLevel.INFO, f"No test configurations found in {flavor_dir.relative_to(inp_path)}", indent=1)
            return 0, 0
        
        console.print()
        console.print(
            Panel(
                f"[cyan]Baseline:[/cyan] {baseline_out.relative_to(flavor_dir)}\n"
                f"[cyan]Configurations to compare:[/cyan] {len(config_dirs)}",
                title=f"[bold cyan]Processing {flavor_dir.relative_to(inp_path)}[/bold cyan]",
                expand=False,
                border_style="cyan",
                padding=(0, 2),
            )
        )
        
        # Track results for summary
        results = []
        
        # Generate diffs for each config
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing configurations...", total=len(config_dirs))
            
            for i, config_dir in enumerate(sorted(config_dirs)):
                if i > 0:
                    console.rule(style="dim")
                
                progress.update(task, description=f"[cyan]Testing [bold]{config_dir.name}[/bold]")
                
                # Find .out file in config directory
                test_out_files = list(config_dir.glob("*.out"))
                if not test_out_files:
                    log_message(LogLevel.WARNING, f"{config_dir.name}: No .out file found, skipping", indent=1)
                    results.append((config_dir.name, False, "No .out file found"))
                    progress.advance(task)
                    continue
                
                test_out = test_out_files[0]
                diff_file = config_dir / "diff_baseline_vs_nd_parallelism.log"
                
                # Extract test metrics
                test_metrics = _extract_metrics(test_out)
                
                # Compare metrics
                if test_metrics.loss and test_metrics.grad_norm:
                    test_passed, metrics_summary = _compare_metrics(baseline_metrics, test_metrics, config_dir.name)
                    
                    if test_passed:
                        log_message(LogLevel.TEST_PASS, f"{config_dir.name} - {metrics_summary}", indent=1)
                        results.append((config_dir.name, True, metrics_summary))
                    else:
                        log_message(LogLevel.TEST_FAIL, f"{config_dir.name} - {metrics_summary}", indent=1)
                        results.append((config_dir.name, False, metrics_summary))
                else:
                    log_message(LogLevel.TEST_FAIL, f"{config_dir.name} - Unable to extract metrics", indent=1)
                    results.append((config_dir.name, False, "Unable to extract metrics"))
                
                # Generate diff
                try:
                    success, output = _generate_diff(baseline_out, test_out, diff_file)
                    
                    if success:
                        log_message(LogLevel.INFO, f"Diff between baseline vs HF nd-parallel saved to:", indent=5, dim=True)
                        console.print(f"      [dim]{diff_file}[/dim]")
                    else:
                        log_message(LogLevel.WARNING, f"Failed to generate diff: {output}", indent=5, dim=True)
                        
                except Exception as e:
                    log_message(LogLevel.WARNING, f"Failed to generate diff - {e}", indent=5, dim=True)
                
                progress.advance(task)
        
        console.print()
        # Create summary table
        summary_table = Table(
            title=f"[bold]Summary for {flavor_dir.relative_to(inp_path)}[/bold]",
            show_header=True,
            header_style="bold magenta"
        )
        summary_table.add_column("Configuration", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Metrics", style="dim")
        
        for name, passed, summary in results:
            status = "[bold green]✅ PASS[/bold green]" if passed else "[bold red]❌ FAIL[/bold red]"
            # Truncate summary if too long
            display_summary = summary if len(summary) < 60 else summary[:57] + "..."
            summary_table.add_row(name, status, display_summary)
        
        console.print(summary_table)
        console.print()
        
        passed_count = sum(1 for _, passed, _ in results if passed)
        failed_count = len(results) - passed_count
        
        return passed_count, failed_count

    inp_path = Path(inp_dir)
    
    if not inp_path.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {inp_path}")
        return
    
    console.print(
        Panel(
            "[bold cyan]HuggingFace Integration Test Report Generator[/bold cyan]",
            expand=False,
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()
    
    # Find all directories that contain a baseline (fsdp2_tp1_cp1_pp1) subdirectory
    flavor_dirs = []
    for root, dirs, files in os.walk(inp_path):
        if BASELINE in dirs:
            flavor_dirs.append(Path(root))
    
    # Filter by --only if provided
    if only:
        original_count = len(flavor_dirs)
        flavor_dirs = [
            d for d in flavor_dirs if only in str(d.relative_to(inp_path))
        ]
        log_message(
            LogLevel.INFO,
            f"Filtered from {original_count} to {len(flavor_dirs)} director{'ies' if len(flavor_dirs) != 1 else 'y'} matching '[bold]{only}[/bold]'",
        )

    if not flavor_dirs:
        log_message(LogLevel.ERROR, f"No directories with baseline configuration found under {inp_path}")
        console.print("[yellow]Expected to find directories containing 'fsdp2_tp1_cp1' subdirectory[/yellow]")
        return
    
    log_message(LogLevel.INFO, f"Found {len(flavor_dirs)} model/flavor combination(s) to process:")
    for flavor_dir in flavor_dirs:
        console.print(f"  [cyan]•[/cyan] {flavor_dir.relative_to(inp_path)}")
    
    # Process each flavor directory
    total_passed = 0
    total_failed = 0
    
    for flavor_dir in flavor_dirs:
        passed, failed = _process_flavor_dir(flavor_dir)
        total_passed += passed
        total_failed += failed
    
    # Final summary
    console.print()
    console.print(
        Panel(
            "[bold cyan]Overall Summary[/bold cyan]",
            expand=False,
            border_style="blue",
            padding=(0, 2),
        )
    )
    
    overall_table = Table(show_header=True, header_style="bold magenta")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", justify="right")
    
    total_tests = total_passed + total_failed
    overall_table.add_row("Total Configurations Tested", str(total_tests))
    overall_table.add_row("[green]Passed[/green]", str(total_passed))
    overall_table.add_row("[red]Failed[/red]", str(total_failed))
    
    console.print(overall_table)
    console.print()
    
    if total_failed == 0 and total_tests > 0:
        log_message(LogLevel.SUCCESS, "All tests passed! 🎉")
    elif total_tests > 0:
        log_message(LogLevel.WARNING, f"{total_failed} configuration(s) had test failures")
    
    log_message(LogLevel.SUCCESS, "Diff generation complete!")

if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    create_configs_parser = subparsers.add_parser("create_configs")
    create_configs_parser.add_argument("--model_name", type=str, required=True)
    create_configs_parser.add_argument("--out_dir", type=str, required=True)
    create_configs_parser.add_argument("--flavor", type=str, required=True)

    submit_jobs_parser = subparsers.add_parser("submit_jobs")
    submit_jobs_parser.add_argument("--inp_dir", type=str, required=True)
    submit_jobs_parser.add_argument("--qos", type=str, required=True, choices=["low", "normal", "high", "prod"])
    submit_jobs_parser.add_argument("--only", type=str, default=None, choices=[s.value for s in Status])

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--inp_dir", type=str, required=True)
    report_parser.add_argument("--only", type=str, default=None)

    check_status_parser = subparsers.add_parser("check_status")
    check_status_parser.add_argument("--inp_dir", type=str, required=True)

    args = parser.parse_args()

    if args.action == "create_configs":
        create_configs(args.model_name, args.out_dir, args.flavor)
    elif args.action == "submit_jobs":
        submit_jobs(args.inp_dir, args.qos, args.only)
    elif args.action == "report":
        report(args.inp_dir, args.only)
    elif args.action == "check_status":
        check_status(args.inp_dir)