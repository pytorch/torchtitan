import submitit
import datetime
import yaml
import os


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    executor.update_parameters(
        name="titan", timeout_min=15,
        gpus_per_node=2,
        nodes=1, mem_gb=30, cpus_per_task=10,
        slurm_array_parallelism=10
    )

    jobs = []
    with executor.batch():
        for _ in range(1):
            function = submitit.helpers.CommandFunction([
                'python3', '-m', 'torch.distributed.run',
                '--nproc_per_node', '2',
                '--rdzv_backend', 'c10d',
                '--rdzv_endpoint', 'localhost:0',
                '--local-ranks-filter', '0',
                '--role', 'rank', '--tee', '3',
                'train.py', '--job.config_file', './train_configs/galactica_125m.toml',
            ])
            print(' '.join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
