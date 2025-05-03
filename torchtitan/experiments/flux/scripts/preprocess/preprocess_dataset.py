# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Optional

import torch

from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer
from torchtitan.experiments.flux.model.autoencoder_utils import encode_with_mean_logvar
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.experiments.flux.utils import preprocess_data
from torchtitan.tools.logging import init_logger, logger


def save_preprocessed_data(
    output_path: str,
    file_name: str,
    data_dict: dict[str, torch.Tensor],
    mean: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
) -> int:
    """
    Save the preprocessed data to a json file. Each rank will save its own data to a different file.
    If mean and logvar are provided, they will be included in the saved data.

    Args:
        output_path: Directory to save the file
        file_name: Name of the file
        data_dict: Dictionary containing the preprocessed data tensors
        mean: Optional mean tensor from the autoencoder
        logvar: Optional logvar tensor from the autoencoder

    Returns: the number of samples in the current batch
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, file_name)

    # append to current file
    with open(output_file, "a") as f:
        bsz = data_dict["t5_encodings"].shape[0]
        for sample_id in range(0, bsz):
            sample_data = {}

            # Add regular preprocessed data
            for key in data_dict.keys():
                if key == "id":
                    continue
                # convert from bFloat16 to float32, since json.dumps didn't support bFloat16
                sample_data[key] = (
                    data_dict[key][sample_id].detach().to(torch.float32).cpu().tolist()
                )

            # Add mean, logvar if provided
            if mean is not None:
                sample_data["mean"] = (
                    mean[sample_id].detach().to(torch.float32).cpu().tolist()
                )

            if logvar is not None:
                sample_data["logvar"] = (
                    logvar[sample_id].detach().to(torch.float32).cpu().tolist()
                )

            f.write(json.dumps(sample_data) + "\n")
    return bsz


class FluxPreprocessor(FluxTrainer):
    """
    Reuse the FluxTrainer class to preprocess the dataset, as the preprocessing is part of the
    training process. Reuse the
    """

    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # Overwrite dataloader and set inifite=False
        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=None,
            job_config=job_config,
            infinite=False,
            include_sample_id=True,
        )
        # load componnents, offload the Flux model to save GPU memory
        self.autoencoder.eval().requires_grad_(False)
        del self.model_parts[0]

        self.preprocess_fn = preprocess_data
        self.job_config = job_config

    def generate_empty_encoding(self):
        # First, calculate and save encodings for empty string, which will be used in classifier-free guidance

        t5_tokenizer = FluxTokenizer(
            self.job_config.encoder.t5_encoder,
            max_length=self.job_config.encoder.max_t5_encoding_len,
        )
        clip_tokenizer = FluxTokenizer(
            self.job_config.encoder.clip_encoder,
            max_length=77,
        )
        empty_batch = {
            "t5_tokens": torch.tensor(t5_tokenizer.encode(""))
            .unsqueeze(1)
            .to(self.device),
            "clip_tokens": torch.tensor(clip_tokenizer.encode(""))
            .unsqueeze(1)
            .to(self.device),
        }

        # Process the empty batch
        empty_encodings = self.preprocess_fn(
            device=self.device,
            dtype=self._dtype,
            autoencoder=None,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=empty_batch,
        )

        if torch.distributed.get_rank() == 0:
            output_path = os.path.join(self.job_config.job.dump_folder, "preprocessed")
            if os.path.exists(output_path):
                os.remove(output_path)
            # Save the empty encodings
            save_preprocessed_data(
                output_path=output_path,
                file_name="empty_encodings.json",
                data_dict=empty_encodings,
            )
        logger.info("Preprocessed empty encodings for classifier-free guidance")

    def preprocess(self):
        self.generate_empty_encoding()

        # Then, calculate and save the encodings for the dataset
        data_iterator = iter(self.dataloader)
        preprocessed_sample_cnt = 0

        while True:
            # try:
            #     input_dict, labels = self.next_batch(data_iterator)
            # except StopIteration:  # If the data iterator run out of data
            #     break
            # except Exception as e:
            #     logger.warning(f"Error processing batch: {e}, skipped...")
            #     continue

            input_dict, labels = self.next_batch(data_iterator)

            for k, _ in input_dict.items():
                if k == "id":
                    continue
                input_dict[k] = input_dict[k].to(self.device)
            labels = labels.to(self.device)

            # Extract mean and logvar values
            mean, logvar = encode_with_mean_logvar(self.autoencoder, labels)

            # Save the sample IDs if present in the input_dict
            sample_ids = None
            if "id" in input_dict:
                sample_ids = input_dict["id"]

            # Continue with regular preprocessing
            input_dict["image"] = labels
            processed_dict = self.preprocess_fn(
                device=self.device,
                dtype=self._dtype,
                autoencoder=None,  # We already extracted the mean and logvar from the autoencoder
                clip_encoder=self.clip_encoder,
                t5_encoder=self.t5_encoder,
                batch=input_dict,
            )
            # Add back the sample IDs if they were present
            if sample_ids is not None:
                processed_dict["id"] = sample_ids

            del (
                processed_dict["image"],
                processed_dict["t5_tokens"],
                processed_dict["clip_tokens"],
            )

            # Save all data together
            bsz = save_preprocessed_data(
                output_path=os.path.join(
                    self.job_config.job.dump_folder, "preprocessed"
                ),
                file_name=f"rank_{torch.distributed.get_rank()}_preprocessed_cc12m.json",
                data_dict=processed_dict,
                mean=mean,
                logvar=logvar,
            )

            # log the process of the preprocessor
            preprocessed_sample_cnt += bsz
            logger.info(
                f"Preprocessed {preprocessed_sample_cnt} samples, "
                f"current batch size: {processed_dict['t5_encodings'].shape[0]}"
            )

            if preprocessed_sample_cnt >= 20:
                break


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    preprocessor: Optional[FluxPreprocessor] = None

    try:
        preprocessor = FluxPreprocessor(config)
        preprocessor.preprocess()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
