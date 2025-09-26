import json
import math
import os
import time
from multiprocessing import Queue

import numpy as np
import requests
import torch
import wandb

from torchtitan.config.job_config import JobConfig
from torchtitan.grpo.sglang_handling import send_wait
from torchtitan.tools.logging import logger


def get_dynamic_batch_gas(
    batch_size, gradient_accumulation_steps, seq_len, max_token_len
):
    dynamic_batch_size = (batch_size * seq_len) // max_token_len
    dynamic_batch_size = min(
        dynamic_batch_size,
        batch_size * gradient_accumulation_steps,
    )
    # To nearest power of 2 of dynamic_batch_size so we have clean splits with gradient accumulation
    dynamic_batch_size = dynamic_batch_size // batch_size
    dynamic_batch_size = (2 ** math.floor(math.log2(dynamic_batch_size))) * batch_size
    dynamic_grad_accum_size = (
        batch_size * gradient_accumulation_steps
    ) // dynamic_batch_size
    return dynamic_batch_size, dynamic_grad_accum_size


def pad_data_to_good_offset(
    data, cp_degree, dp_degree, scale_adv_by_len=True, num_microbatches=1
):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64 * cp_degree, since it's split along context parallel dimensions
    # TODO: see if FSDP affects this as well
    good_multiple = 64 * cp_degree
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = (
            max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
            max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    rewards = list()
    masks = list()
    lengths = list()
    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        if scale_adv_by_len:
            unmasked_lengths = list()
            for i in range(len(item["masks"])):
                unmasked_lengths.append(
                    (np.array(item["masks"][i]) != -100).astype(float).sum()
                )
            unmasked_lengths = np.array(unmasked_lengths)
            scores = scores * unmasked_lengths
        # check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0
        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple))
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            item["masks"][i] = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["masks"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["masks"][i] = (item["masks"][i] != -100).astype(int)
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            rewards.append(item["scores"][i])
            masks.append(item["masks"][i][1:])
    # sort into 4 buckets...
    raw_items = [
        {
            "input_id": input_id,
            "label": label,
            "mask": mask,
            "reward": reward,
            "length": length,
        }
        for (input_id, label, mask, reward, length) in zip(
            input_ids, labels, masks, rewards, lengths
        )
    ]
    sorted_items = sorted(raw_items, key=lambda x: x["length"], reverse=True)
    # Create dp_degree * num_microbatches buckets for balanced distribution
    total_buckets = dp_degree * num_microbatches
    buckets = [[] for _ in range(total_buckets)]
    out_lens = list()
    for i in range(len(sorted_items)):
        buckets[i % total_buckets].append(sorted_items[i])
    # merge... ensuring microbatches are contiguous for each DP rank
    items = []
    for dp_idx in range(dp_degree):
        for mb_idx in range(num_microbatches):
            bucket_idx = dp_idx * num_microbatches + mb_idx
            items.extend(buckets[bucket_idx])
    return (
        max_token_len,
        [x["input_id"] for x in items],
        [x["label"] for x in items],
        [x["reward"] for x in items],
        [x["mask"] for x in items],
        [x["length"] for x in items],
    )


def prep_data(
    data,
    cp_degree,
    dp_degree,
    batch_size,
    gradient_accumulation_steps,
    seq_len,
    scale_adv_by_len=True,
    num_microbatches=1,
):

    # Now prepare the batch
    batches = list()

    max_token_len, input_ids, labels, rewards, masks, lengths = pad_data_to_good_offset(
        data,
        cp_degree,
        dp_degree,
        scale_adv_by_len=scale_adv_by_len,
        num_microbatches=num_microbatches,
    )

    dynamic_batch_size, dynamic_grad_accum_size = get_dynamic_batch_gas(
        batch_size, gradient_accumulation_steps, seq_len, max_token_len
    )
    # now allocate to new batch/grad sizes
    for i in range(dynamic_grad_accum_size * dp_degree):
        start = i * dynamic_batch_size
        end = (i + 1) * dynamic_batch_size
        batches.append(
            (
                np.array(input_ids[start:end]),
                np.array(labels[start:end]),
                np.array(masks[start:end]),
                np.array(rewards[start:end]),
            )
        )
    return batches, max_token_len, dynamic_batch_size, dynamic_grad_accum_size, lengths


def prep_empty_data_matricies(
    max_token_len, batch_size, gradient_accumulation_steps, seq_len, dp_degree
):
    dynamic_batch_size = batch_size * seq_len // max_token_len
    dynamic_batch_size = min(
        dynamic_batch_size,
        batch_size * gradient_accumulation_steps,
    )
    # To nearest power of 2 of dynamic_batch_size so we have clean splits with gradient accumulation
    dynamic_batch_size = dynamic_batch_size // batch_size
    dynamic_batch_size = (2 ** math.floor(math.log2(dynamic_batch_size))) * batch_size
    dynamic_grad_accum_size = (
        batch_size * gradient_accumulation_steps
    ) // dynamic_batch_size
    batches = [
        [
            np.zeros((dynamic_batch_size, max_token_len), dtype=np.int64),
            np.zeros((dynamic_batch_size, max_token_len), dtype=np.int64),
            np.zeros((dynamic_batch_size, max_token_len), dtype=np.int64),
            np.zeros(dynamic_batch_size, dtype=np.float32),
        ]
        for _ in range(dynamic_grad_accum_size * dp_degree)
    ]
    return batches, max_token_len, dynamic_batch_size, dynamic_grad_accum_size


def data_worker(
    queue: Queue,
    server_url,
    cp_degree,
    dp_degree,
    batch_size,
    gradient_accumulation_steps,
    seq_len,
    grpo_scale_adv_by_len,
):
    while True:
        if not queue.empty():
            # Only have one batch in the queue at a time, otherwise we can overshoot and get too far off policy
            time.sleep(1)
            continue
        data = requests.get(f"{server_url}/batch").json()
        if data["batch"] is not None:
            # Save the batch
            with open("temp.json", "w") as f:
                json.dump(data, f)
            (
                batches,
                max_token_len,
                dynamic_batch_size,
                dynamic_grad_accum_size,
                data_lens,
            ) = prep_data(
                data,
                cp_degree,
                dp_degree,
                batch_size,
                gradient_accumulation_steps,
                seq_len,
                scale_adv_by_len=grpo_scale_adv_by_len,
            )
            queue.put(
                (
                    batches,
                    max_token_len,
                    dynamic_batch_size,
                    dynamic_grad_accum_size,
                    data_lens,
                )
            )


class OnlineDataHandler:
    def __init__(self):
        if int(os.environ.get("SLURM_NODEID", "0")) == 0:
            self.server_url = "http://localhost:8000"
        else:
            self.server_url = f'http://{os.environ["head_node_ip"]}:8000'
        if torch.distributed.get_rank() == 0:
            self.queue = Queue()
        else:
            self.queue = None

    def register_atropos(
        self,
        job_config: JobConfig,
        step: int,
        global_batch_size: int,
    ):
        """
        Registers a training job with the Atropos service. This function sends a
        POST request to register the current training job configuration, allowing
        Atropos to monitor and manage it.

        Parameters:
        - job_config (JobConfig): Configuration containing details for the training job.
        - step (int): The starting step of the training process.
        - global_batch_size (int): Total batch size used for training globally
                                   across all workers.
        """
        if torch.distributed.get_rank() == 0:
            requests.post(
                f"{self.server_url}/register",
                json={
                    "wandb_group": wandb.run.group,
                    "wandb_project": wandb.run.project,
                    "batch_size": global_batch_size,
                    "max_token_len": job_config.training.seq_len,
                    "starting_step": step,
                    "checkpoint_dir": job_config.checkpoint.folder,
                    "save_checkpoint_interval": job_config.checkpoint.interval,
                    "num_steps": job_config.training.steps,
                },
            )
            # Startup data process
            # grad_accum_size = job_config.training.global_batch_size // (
            #         job_config.training.local_batch_size * dp_degree
            # )
            # grad_accum_size = max(1, grad_accum_size)
            # data_process = Process(target=data_worker, args=(
            #     self.queue,
            #     self.server_url,
            #     cp_degree,
            #     dp_degree,
            #     job_config.training.local_batch_size,
            #     grad_accum_size,
            #     job_config.training.seq_len,
            #     job_config.grpo.scale_adv_by_len
            # ))
            # data_process.start()

    def data_handling(
        self,
        sglang_gloo_group,
        cp_degree,
        dp_degree,
        dp_replicate_rank,
        device,
        job_config: JobConfig,
        step: int,
    ):
        """
        Handles the data for the current training step from atropos.

        Args:
            sglang_gloo_group: The sglang group to use for all communication
            cp_degree: The number of context parallel replicas
            dp_degree: The number of data parallel replicas
            dp_replicate_rank: The rank within the data parallel group
            device: The device to place tensors on
            job_config: Configuration containing training parameters and settings
            step: The current training step

        Returns:
            tuple: A tuple containing:
                - batches: List of prepared data batches
                - max_token_len: Maximum token length for the current batch
                - dynamic_batch_size: Calculated batch size based on sequence length
                - dynamic_grad_accum_size: Number of gradient accumulation steps
                - data_lens: List of sequence lengths for each sample
        """
        flag = torch.tensor(0).to(device)
        grad_accum_size = job_config.training.global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        grad_accum_size = max(1, grad_accum_size)
        while True:
            if torch.distributed.get_rank() == 0:
                # if not self.queue.empty():
                #     (
                #         batches,
                #         max_token_len,
                #         dynamic_batch_size,
                #         dynamic_grad_accum_size,
                #         data_lens,
                #     ) = self.queue.get()
                data = requests.get(f"{self.server_url}/batch").json()
                if data["batch"] is not None:
                    logger.debug("Rx'd batch from server...")
                    # Save the batch
                    with open("temp.json", "w") as f:
                        json.dump(data, f)
                    (
                        batches,
                        max_token_len,
                        dynamic_batch_size,
                        dynamic_grad_accum_size,
                        data_lens,
                    ) = prep_data(
                        data,
                        cp_degree,
                        dp_degree,
                        job_config.training.local_batch_size,
                        grad_accum_size,
                        job_config.training.seq_len,
                        scale_adv_by_len=job_config.grpo.scale_adv_by_len,
                    )
                    flag = flag + 1
                    torch.distributed.broadcast(flag, 0)
                    if dp_replicate_rank == 0:
                        send_wait(sglang_gloo_group)
                    max_token_len = torch.tensor(max_token_len).to(device)
                    torch.distributed.all_reduce(max_token_len)
                    # back to int
                    max_token_len = max_token_len.item()
                    # distribute the lengths
                    torch.distributed.broadcast_object_list(data_lens, 0)
                    # now broadcast the batch
                    torch.distributed.broadcast_object_list(batches, 0)
                    break
                else:
                    logger.debug("No batch yet, retrying...")
                    torch.distributed.broadcast(flag, 0)
                    if dp_replicate_rank == 0:
                        send_wait(sglang_gloo_group)
            else:
                logger.debug("Waiting for batch from server...")
                torch.distributed.broadcast(flag, 0)
                if dp_replicate_rank == 0:
                    send_wait(sglang_gloo_group)
                if flag.item() > 0:
                    # Got the batch
                    max_token_len = torch.tensor(0).to(device)
                    torch.distributed.all_reduce(max_token_len)
                    # back to int
                    max_token_len = max_token_len.item()
                    data_lens = [
                        0
                        for _ in range(
                            job_config.training.local_batch_size
                            * grad_accum_size
                            * dp_degree
                        )
                    ]
                    torch.distributed.broadcast_object_list(data_lens, 0)
                    (
                        batches,
                        max_token_len,
                        dynamic_batch_size,
                        dynamic_grad_accum_size,
                    ) = prep_empty_data_matricies(
                        max_token_len,
                        job_config.training.local_batch_size,
                        grad_accum_size,
                        job_config.training.seq_len,
                        dp_degree,
                    )
                    # now get the batch
                    torch.distributed.broadcast_object_list(batches, 0)
                    break
            time.sleep(1)
        # Now to check data...
        all_good = 0
        comp_bs = len(batches[0][0])
        try:
            for batch in batches:
                for j in range(len(batch[0])):
                    assert (
                        len(batch[0][j]) == max_token_len
                    ), f"Token lengths indx {j} don't match max token length!"
                    assert (
                        len(batch[1][j]) == max_token_len
                    ), f"Label lengths indx {j} don't match max token length!"
                    assert (
                        len(batch[2][j]) == max_token_len
                    ), f"Mask lengths indx {j} don't match max token length!"
        except AssertionError as e:
            filename = (
                f"batch_step_{step + 1}_rank_{torch.distributed.get_rank()}_error.json"
            )
            logger.error(f"Data mismatch! Saving temp to {filename} and continuing...")
            logger.error(f"Mismatch error: {e}")
            with open(filename, "w") as f:
                json.dump(data, f)
            all_good += 1
        # check to see if every all_good is 0...
        all_good = torch.tensor(all_good).to(device)
        torch.distributed.all_reduce(all_good)
        if all_good.item() > 0:
            logger.error(
                f"data error on step {step + 1}, check for json file for the batch data. skipping..."
            )
            raise AssertionError("Data mismatch!")
        return (
            batches,
            max_token_len,
            dynamic_batch_size,
            dynamic_grad_accum_size,
            data_lens,
        )


if __name__ == "__main__":
    print(
        pad_data_to_good_offset(
            {
                "batch": [
                    {
                        "tokens": [[1, 2, 3], [4, 5, 6]],
                        "masks": [[-100, 2, 3], [-100, 5, 6]],
                        "scores": [-1, 1.0],
                        "overrides": None,
                    }
                ]
            },
            cp_degree=1,
            dp_degree=1,
            scale_adv_by_len=True,
            num_microbatches=1,
        )
    )
