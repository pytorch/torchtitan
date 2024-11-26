import os
from typing import Any, Dict, Sequence

from torchtitan.metrics import MetricRetriever

metrics: Dict[str, Dict[int, Dict[str, Any]]] = {}

run_id = 0
dump_dir = f"test_out/my_example"
log_dir = os.path.join(dump_dir, "tb", str(run_id))
print(log_dir)
metric_retriever = MetricRetriever(log_dir)

metrics[f"Run ID {run_id}"] = metric_retriever.get_metrics()


def print_metrics(
    metrics: Dict[str, Dict[int, Dict[str, Any]]],
    filter_keys=[
        "wps",
        "mfu(%)",
        "memory/max_active(GiB)",
        "memory/max_active(%)",
        "memory/max_reserved(%)",
        "loss_metrics/global_avg_loss",
        "loss_metrics/global_max_loss",
    ],
) -> None:
    for run_id, all_step_metrics in metrics.items():
        print("=" * 100)
        print(run_id)
        print("=" * 100)
        if all_step_metrics:
            last_step = next(reversed(all_step_metrics))
            last_step_metrics = all_step_metrics[last_step]
            # Print the column headers
            if filter_keys:
                filtered_keys = [key for key in filter_keys if key in last_step_metrics]
            else:
                filtered_keys = list(last_step_metrics.keys())

            max_key_length = max(len(key) for key in filtered_keys)
            # Add an empty header for the run_id column
            header_row = " | ".join(
                [" " * 10] + [f"{key.ljust(max_key_length)}" for key in filtered_keys]
            )
            print(header_row)
            print("-" * len(header_row))
            # Print the run_id and the values
            value_row = " | ".join(
                [f"{run_id:10}"]
                + [
                    f"{str(last_step_metrics[key]).ljust(max_key_length)}"
                    for key in filtered_keys
                ]
            )
            print(value_row)


print_metrics(metrics)
