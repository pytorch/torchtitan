import toml
from argparse import ArgumentParser
from pathlib import Path
import re
import os
import subprocess
from enum import Enum
from jinja2 import Template


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
                    |_baseline_fsdp2/
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
                        |_ diff_baseline_vs_nd_parallelism.log
                    |_ fsdp2_tp2_cp2_pp1/
                        |_ config.toml
                        |_ nd_parallelism.slurm
                        |_ nd_parallelism.log
                        |_ diff_baseline_vs_nd_parallelism.log
                    |_ fsdp2_tp2_cp2_pp2/
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

    parallelism_configs = [
        "fsdp2_tp1_cp1_pp1", # baseline
        "fsdp2_tp2_cp1_pp1",
        "fsdp2_tp1_cp1_pp2",
        "fsdp2_tp1_cp2_pp1",
        "fsdp2_tp1_cp2_pp2",
        "fsdp2_tp2_cp2_pp1",
        "fsdp2_tp2_cp2_pp2",
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
    seed_config["model"]["hf_assets_path"] = str(seed_checkpoint_dir / Path(model_name).name)
    seed_config["model"]["tokenizer_path"] = str(seed_checkpoint_dir / Path(model_name).name)
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
        iter_config["parallelism"]["pipeline_parallel_schedule"] = "1F1B"
        iter_config["model"]["hf_assets_path"] = str(seed_checkpoint_dir / Path(model_name).name)

        config_path = pc_dir / "config.toml"
        with open(config_path, "w") as f:
            toml.dump(iter_config, f)
        print(f"Created {config_path}")
        _create_slurm_script(
            iter_config,
            config_path,
            pc_dir / "nd_parallelism.slurm",
            pc,
            initial_load_path=str(seed_checkpoint_dir / "step-0"),
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
    def __init__(self, root_path: str, qos: str) -> None:
        self.root_path = root_path
        self.name = os.path.basename(root_path)
        if self.name == os.path.basename(os.path.normpath(args.inp_dir)):
            self.name = "baseline_fsdp2"
            self.config = os.path.join(root_path, "baseline_fsdp2_config.toml")
            self.slurm_script = os.path.join(root_path, "baseline_fsdp2.slurm")
        else:
            self.config = os.path.join(root_path, "config.toml")
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

        self.job_lists = [Job(job_path, qos) for job_path in jobs_directory_paths]

    def keep_only_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status == status]

    def filter_out_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status != status]

    def check_status(self):
        status_files = [os.path.join(job.root_path, "status.txt") for job in self.job_lists]

        status_counts = {status.value: 0 for status in Status}

        for status_file in status_files:
            with open(status_file, "r") as f:
                status = f.read().strip()
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    raise ValueError(f"Invalid status: {status}")

        total = sum(status_counts.values())

        print(f"{'Status':<10} | {'Count':<6}")
        print(f"{'-'*10}-|-{'-'*6}")
        for status, count in status_counts.items():
            print(f"{status.capitalize():<10} | {count:<6}")

        print(f"{'-'*10}-|-{'-'*6}")
        print(f"{'Total':<10} | {total:<6}")


def submit_jobs(inp_dir, qos, only: str = None, seed_checkpoint: str = None):
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


def report(inp_dir: str):
    scheduler = Scheduler(inp_dir, qos="N/A")
    scheduler.check_status()


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    create_configs_parser = subparsers.add_parser("create_configs")
    create_configs_parser.add_argument("--model_name", type=str, required=True)
    create_configs_parser.add_argument("--out_dir", type=str, required=True)
    create_configs_parser.add_argument("--flavor", type=str, required=True)
    submit_jobs_parser = subparsers.add_parser("submit_jobs")
    submit_jobs_parser.add_argument("--inp_dir", type=str, required=True)
    submit_jobs_parser.add_argument("--seed_checkpoint", type=str, default=None)
    submit_jobs_parser.add_argument("--qos", type=str, required=True, choices=["low", "normal", "high", "prod"])
    submit_jobs_parser.add_argument("--only", type=str, default=None, choices=[s.value for s in Status])

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--inp_dir", type=str, required=True)

    args = parser.parse_args()

    if args.action == "create_configs":
        create_configs(args.model_name, args.out_dir, args.flavor)
    elif args.action == "submit_jobs":
        submit_jobs(args.inp_dir, args.qos, args.only, args.seed_checkpoint)
    elif args.action == "report":
        report(args.inp_dir)