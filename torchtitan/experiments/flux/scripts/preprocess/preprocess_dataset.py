import json
import os
from typing import Optional

import torch

from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.flux import flux_configs
from torchtitan.experiments.flux.dataset.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.model.autoencoder import load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.experiments.flux.utils import preprocess_data
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


def save_preprocessed_data(output_path: str, data_dict: dict[str, torch.Tensor]) -> int:
    """
    Save the preprocessed data to a json file. Each rank will save its own data to a different file.

    Returns: the number of samples in the current batch
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = f"rank_{torch.distributed.get_rank()}_preprocessed_cc12m.json"
    output_file = os.path.join(output_path, file_name)

    # append to current file
    with open(output_file, "a") as f:
        bsz = data_dict["img_encodings"].shape[0]
        for sample_id in range(0, bsz):
            sample_data = {}
            for key in ["img_encodings", "clip_encodings", "t5_encodings"]:
                # convert from bFloat16 to float32, since json.dumps didn't support bFloat16
                sample_data[key] = (
                    data_dict[key][sample_id].detach().to(torch.float32).cpu().tolist()
                )

            f.write(json.dumps(sample_data) + "\n")
    return bsz


class FluxPreprocessor(FluxTrainer):
    def __init__(self, job_config: JobConfig):
        """
        Reuse the FluxTrainer class to preprocess the dataset, as the preprocessing is part of the
        training process. Reuse the
        """
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
        )
        # load componnents
        self.autoencoder.eval().requires_grad_(False)
        self.model_parts[0].to_device("cpu")

        self.preprocess_fn = preprocess_data
        self.job_config = job_config

    def preprocess(self):
        data_iterator = iter(self.dataloader)
        preprocessed_sample_cnt = 0

        while True:
            try:
                input_dict, labels = self.next_batch(data_iterator)
                for k, _ in input_dict.items():
                    input_dict[k] = input_dict[k].to(self.device)
                labels = labels.to(self.device)

                input_dict["image"] = labels
                input_dict = self.preprocess_fn(
                    device=self.device,
                    dtype=self._dtype,
                    autoencoder=self.autoencoder,
                    clip_encoder=self.clip_encoder,
                    t5_encoder=self.t5_encoder,
                    batch=input_dict,
                )

                print(input_dict["img_encodings"].shape)

                bsz = save_preprocessed_data(
                    self.job_config.job.dump_folder, input_dict
                )

                # log the process of the preprocessor
                preprocessed_sample_cnt += bsz
                logger.info(
                    f"Preprocessed {preprocessed_sample_cnt} samples, "
                    f"current batch size: {input_dict['img_encodings'].shape[0]}"
                )

            except Exception:  # Add error handling if next_batch load failed
                logger.warning("Skip error batch in preprocessing")

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
