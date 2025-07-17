# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import shutil

import numpy as np

# For more memory-efficient streaming approach
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.elastic.multiprocessing.errors import record

# Import from the existing codebase
from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux.dataset.tokenizer import build_flux_tokenizer
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.experiments.flux.utils import preprocess_data
from tqdm import tqdm


def merge_datasets(path):
    dataset = Dataset.from_parquet(os.path.join(path + "_temp", "*"), num_proc=16)
    dataset.save_to_disk(path, num_proc=8)
    shutil.rmtree(path + "_temp")


def process_with_streaming(trainer, global_id: int):
    print(f"Reading data from {trainer.job_config.training.dataset_path}")
    temp_path = trainer.job_config.preprocessing.output_dataset_path + "_temp"
    print(f"Writing data temporarily to {temp_path}")

    schema = pa.schema(
        [
            ("__key__", pa.string()),
            ("t5_encodings", pa.binary()),
            ("clip_encodings", pa.binary()),
            ("mean", pa.binary()),
            ("logvar", pa.binary()),
            ("timestep", pa.int32()),
        ]
    )

    # Configuration for file splitting
    SAMPLES_PER_FILE = 10000  # Adjust this to control file size
    batch_idx = 0
    total_samples = 0
    file_idx = 0
    current_file_samples = 0
    writer = None

    try:
        with torch.no_grad():
            for inputs, labels in tqdm(
                trainer.batch_generator(trainer.dataloader), desc=f"Rank {global_id}"
            ):
                inputs["image"] = labels
                inputs = preprocess_data(
                    device=trainer.device,
                    dtype=trainer._dtype,
                    autoencoder=trainer.autoencoder,
                    clip_encoder=trainer.clip_encoder,
                    t5_encoder=trainer.t5_encoder,
                    batch=inputs,
                    return_mean_logvar=True,
                )

                # Check if we need to start a new file
                if writer is None or current_file_samples >= SAMPLES_PER_FILE:
                    # Close current writer if exists
                    if writer is not None:
                        writer.close()
                        print(
                            f"Rank {global_id}: Closed file {file_idx-1} with {current_file_samples} samples"
                        )

                    # Start new file
                    parquet_path = (
                        f"{temp_path}/data_{global_id}_{file_idx:04d}.parquet"
                    )
                    writer = pq.ParquetWriter(parquet_path, schema)
                    current_file_samples = 0
                    print(
                        f"Rank {global_id}: Started new file {file_idx}: {parquet_path}"
                    )
                    file_idx += 1

                # Convert to arrow format and write immediately
                batch_data = []

                # Batch CPU transfers to reduce sync points
                cpu_tensors = {
                    "t5_encodings": inputs["t5_encodings"].to(
                        device="cpu", dtype=torch.bfloat16
                    ),
                    "clip_encodings": inputs["clip_encodings"].to(
                        device="cpu", dtype=torch.bfloat16
                    ),
                    "mean": inputs["mean"].to(device="cpu", dtype=torch.bfloat16),
                    "logvar": inputs["logvar"].to(device="cpu", dtype=torch.bfloat16),
                }

                if "timestep" in inputs:
                    cpu_tensors["timestep"] = inputs["timestep"]

                for i in range(len(inputs["id"])):
                    single_sample = {
                        "__key__": inputs["id"][i],
                        "t5_encodings": serialize_numpy_array(
                            cpu_tensors["t5_encodings"][i]
                        ),
                        "clip_encodings": serialize_numpy_array(
                            cpu_tensors["clip_encodings"][i]
                        ),
                        "mean": serialize_numpy_array(cpu_tensors["mean"][i]),
                        "logvar": serialize_numpy_array(cpu_tensors["logvar"][i]),
                    }
                    if "timestep" in inputs:
                        single_sample["timestep"] = cpu_tensors["timestep"][i]
                    batch_data.append(single_sample)

                # Write batch to current parquet file
                batch_table = pa.Table.from_pylist(batch_data, schema=schema)
                writer.write_table(batch_table)

                # Update counters
                batch_size = len(batch_data)
                total_samples += batch_size
                current_file_samples += batch_size

                # Clean up memory
                del inputs, batch_data, batch_table, cpu_tensors

                batch_idx += 1

    finally:
        # Close the final writer
        if writer is not None:
            writer.close()
            print(
                f"Rank {global_id}: Closed final file {file_idx-1} with {current_file_samples} samples"
            )

    print(
        f"Rank {global_id}: Saved {total_samples} processed samples across {file_idx} files"
    )
    return total_samples


def serialize_numpy_array(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes preserving shape and dtype.

    Args:
        arr: Input numpy array
    """
    buffer = io.BytesIO()

    # Store bf16 as uint16 view in numpy (preserves bf16 precision, minimal overhead)
    # Convert bf16 to uint16 view for numpy storage
    bf16_as_uint16 = arr.view(torch.uint16).numpy()
    np.save(buffer, bf16_as_uint16)

    return buffer.getvalue()


def deserialize_numpy_array(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes.

    Args:
        data: Serialized bytes
    """
    buffer = io.BytesIO(data)

    # Load uint16 view and convert back to bf16
    uint16_data = np.load(buffer)
    tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
    return tensor.numpy()


def deserialize_preprocessed_example(example):
    """
    Utility function to deserialize arrays from preprocessed dataset during training.

    Usage:
        dataset = load_from_disk("/path/to/preprocessed/dataset")
        dataset = dataset.map(deserialize_preprocessed_example)
    """
    example["t5_encodings"] = deserialize_numpy_array(example["t5_encodings"])
    example["clip_encodings"] = deserialize_numpy_array(example["clip_encodings"])
    example["mean"] = deserialize_numpy_array(example["mean"])
    example["logvar"] = deserialize_numpy_array(example["logvar"])
    if "timestep" in example:
        example["timestep"] = example["timestep"]
    return example


@record
def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer = FluxTrainer(config)
    del trainer.model_parts
    global_id = int(os.environ["RANK"])
    trainer.dataloader.dataset.infinite = False

    t5_tokenizer, clip_tokenizer = build_flux_tokenizer(trainer.job_config)
    trainer.t5_tokenizer = t5_tokenizer
    trainer.clip_tokenizer = clip_tokenizer
    if global_id == 0:
        os.makedirs(config.preprocessing.output_dataset_path, exist_ok=False)
        os.makedirs(config.preprocessing.output_dataset_path + "_temp", exist_ok=False)

    try:
        print(f"Rank {global_id}: Starting preprocessing...")
        dist.barrier()
        process_with_streaming(trainer, global_id)

        # Synchronize all processes after preprocessing
        if dist.is_initialized():
            print(f"Rank {global_id}: Preprocessing completed, synchronizing...")

            # Set longer timeout for barrier - preprocessing can take hours on large datasets
            # and we need all processes to complete before merging
            old_timeout = os.environ.get("NCCL_TIMEOUT_MS", None)
            os.environ["NCCL_TIMEOUT_MS"] = str(4 * 60 * 60 * 1000)  # 4 hours

            try:
                dist.barrier()
                print(f"Rank {global_id}: Preprocessing sync completed!")
            finally:
                # Restore original timeout
                if old_timeout is not None:
                    os.environ["NCCL_TIMEOUT_MS"] = old_timeout
                else:
                    # Remove the timeout if it wasn't set originally
                    os.environ.pop("NCCL_TIMEOUT_MS", None)
        else:
            print(f"Rank {global_id}: Distributed not initialized, skipping sync...")

        # Only main process does the merge (no GPU needed)
        if global_id == 0:
            print("Main process: Starting dataset merge...")
            merge_datasets(config.preprocessing.output_dataset_path)
            print("Main process: Dataset merge completed successfully!")
        else:
            print(f"Rank {global_id}: Exiting after preprocessing")

    except Exception as e:
        print(f"Rank {global_id}: Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
