import submitit
import datetime
import yaml
import os


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    n_gpus = 8
    executor.update_parameters(
        name="titan", timeout_min=3 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1, mem_gb=80, cpus_per_task=n_gpus * 2,
        slurm_additional_parameters={
            "partition": "h100"
        }
    )

    jobs = []
    with executor.batch():
        for _ in range(1):
            # train_config = './train_configs/chemlactica_125m.toml'
            train_config = './train_configs/chemlactica_1.3b.toml'
            # train_config = './train_configs/llama3_8b.toml'
            # train_config = './train_configs/debug_model.toml'
            function = submitit.helpers.CommandFunction([
                'python3', '-m', 'torch.distributed.run',
                '--nproc_per_node', f'{n_gpus}',
                '--rdzv_backend', 'c10d',
                '--rdzv_endpoint', 'localhost:0',
                '--local-ranks-filter', '0',
                '--role', 'rank', '--tee', '3',
                'train.py',
                '--job.config_file', train_config,
            ])
            print(' '.join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
