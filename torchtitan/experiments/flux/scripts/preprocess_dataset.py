import json
import os
from typing import Optional

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.flux import flux_configs
from torchtitan.experiments.flux.dataset.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.model.autoencoder import load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.utils import preprocess_flux_data
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


def save_preprocessed_data(
    output_path: str, dp_rank: int, data_dict: dict[str, torch.Tensor]
):
    """
    Save the preprocessed data to a json file. Each rank will save its own data to a different file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = f"rank_{dp_rank}_preprocessed_cc12m.json"
    output_file = os.path.join(output_path, file_name)

    # append to current file
    with open(output_file, "a") as f:
        bsz, _, _ = data_dict["img_encodings"].shape
        for sample_id in range(0, bsz):
            sample_data = {
                "img_encoding": data_dict["img_encodings"][sample_id],
                "clip_encoding": data_dict["clip_encodings"][sample_id],
                "t5_encoding": data_dict["clip_tokens"][sample_id],
            }
            f.write(json.dumps(sample_data))


class FluxPreprocessor:
    def __init__(self, job_config: JobConfig):
        """
        Reuse the FluxTrainer class to preprocess the dataset, as the preprocessing is part of the
        training process. Reuse the
        """

        # initialize distributed
        _, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        world_size = int(os.environ["WORLD_SIZE"])

        # TODO: change to passing from command line
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=1,
            tp=1,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        dist_utils.init_distributed(job_config)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # set random seed: Each FSDP rank has same
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )

        self._dtype = torch.bfloat16

        # load componnents
        self.autoencoder = load_ae(
            job_config.encoder.auto_encoder_path,
            flux_configs["flux-schnell"].autoencoder_params,
            device=self.device,
            dtype=self._dtype,
        )
        self.clip_encoder = FluxEmbedder(version=job_config.encoder.clip_encoder).to(
            device=self.device, dtype=self._dtype
        )
        self.t5_encoder = FluxEmbedder(version=job_config.encoder.t5_encoder).to(
            device=self.device, dtype=self._dtype
        )
        self.dataloader = build_flux_dataloader(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=None,
            job_config=job_config,
        )

        self.preprocess_fn = preprocess_flux_data
        self.job_config = job_config

    def preprocess(self):
        data_iterator = iter(self.dataloader)
        preprocessed_sample_cnt = 0
        for batch in data_iterator:
            input_dict, labels = batch
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

            dp_rank = (
                self.world_mesh["dp"].get_local_rank()
                if self.parallel_dims.dp_enabled
                else 0
            )
            save_preprocessed_data(self.job_config.job.dump_folder, dp_rank, input_dict)

            # log the process of the preprocessor
            bsz = input_dict["img_encodings"].shape[0]
            preprocessed_sample_cnt += bsz
            logger.info(
                f"Preprocessed {preprocessed_sample_cnt} samples, "
                f"current batch size: {input_dict['img_encodings'].shape[0]}"
            )


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.maybe_add_custom_args()
    config.parse_args()
    preprocessor: Optional[FluxPreprocessor] = None

    try:
        preprocessor = FluxPreprocessor(config)
        preprocessor.preprocess()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
