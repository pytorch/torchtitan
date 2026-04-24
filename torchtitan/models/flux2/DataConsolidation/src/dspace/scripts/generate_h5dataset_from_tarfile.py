from pathlib import Path
import tarfile
import json
from typing import Dict, Optional, Tuple
import io
from tqdm import tqdm

import numpy as np

from PIL import Image

import fire
import h5py


def extract_file(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    captions_dict: Dict,
    output_img_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    with tar.extractfile(member) as f:
        filename = member.name.split("/")[-1]
        img_info_dict = captions_dict[filename]
        data = f.read()

        if output_img_size:
            img = Image.open(io.BytesIO(data)).resize(output_img_size)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                data = np.frombuffer(output.getvalue(), dtype=np.uint8)

        return {
            "filename": filename,
            "bytes": data,
            "caption": img_info_dict["txt"],
            "drive_sequence_id": "_".join(
                [img_info_dict["drive_id"], img_info_dict["sequence_id"]]
            ),
            "cam_id": img_info_dict["cam_id"],
        }


class CommandLineConfig:
    def __init__(
        self,
        tar_path: str | Path,
        caption_root: str | Path = "/p/scratch/nxtaim-1/proprietary/dspace/captions/",
        output_h5_root: str | Path = "/p/scratch/nxtaim-1/proprietary/dspace/h5_files",
        output_img_size: Optional[Tuple[int, int] | int] = None,
    ):
        if isinstance(tar_path, str):
            tar_path = Path(tar_path)
        if not tar_path.exists():
            raise FileNotFoundError

        drive_id = tar_path.stem

        if isinstance(output_h5_root, str):
            output_h5_root = Path(output_h5_root)
        if not output_h5_root.exists():
            raise FileNotFoundError(f"No such output directory: {output_h5_root}.")

        if isinstance(caption_root, str):
            caption_root = Path(caption_root)
        if not caption_root.exists():
            raise FileNotFoundError(f"No such caption directory: {caption_root}.")
        if not (caption_root / (drive_id + ".json")).exists():
            raise FileNotFoundError(
                f"Caption json {(caption_root / (drive_id + '.json'))} not found"
            )

        if isinstance(output_img_size, int):
            output_img_size = (output_img_size, output_img_size)

        output_h5 = output_h5_root / (drive_id + ".h5")
        if output_h5.exists():
            raise FileExistsError

        with tarfile.open(tar_path, "r") as th:

            members = th.getmembers()
            # filter for images only
            members = [m for m in members if "sequences" in m.name]
            members = [m for m in members if m.name.endswith(".png")]

        total_len = len(members)

        with h5py.File(output_h5, "w") as hdf:
            dt_img = h5py.vlen_dtype(np.dtype(np.uint8))
            dt_caption = h5py.string_dtype(encoding="utf-8")
            dt_str = h5py.string_dtype(encoding="utf-8")

            dset_image = hdf.create_dataset(
                "image_bytes", shape=(total_len,), dtype=dt_img
            )
            dset_captions = hdf.create_dataset(
                "captions", shape=(total_len,), dtype=dt_caption
            )
            dset_filenames = hdf.create_dataset(
                "filenames", shape=(total_len,), dtype=dt_str
            )
            dset_drive_sequence_id = hdf.create_dataset(
                "drive_sequence_id", shape=(total_len,), dtype=dt_str
            )

            idx = 0

            with open(caption_root / (drive_id + ".json"), "r") as fh:
                captions_dict = json.load(fh)
            with tarfile.open(tar_path) as th:
                for member in tqdm(members):
                    data_dict = extract_file(
                        th,
                        member,
                        captions_dict=captions_dict,
                        output_img_size=output_img_size,
                    )
                    if data_dict is not None:
                        dset_image[idx] = np.array(
                            list(data_dict["bytes"]), dtype=np.uint8
                        )
                        dset_captions[idx] = data_dict["caption"]
                        dset_filenames[idx] = data_dict["filename"]
                        dset_drive_sequence_id[idx] = data_dict["drive_sequence_id"]
                        idx += 1


if __name__ == "__main__":
    """
    Generates an hdf5 file from a tarfile in dSPACE format.
    The filename of the generated h5 file is identical to tar filename.
    Arguments:
        - tar_path: path to the tarfile which should be converted
        - output_h5_root: directory where the generated h5 file is saved
        - output_img_size: Optional, if provided images are resized

    The generated hdf5 file contains four datasets:
        - 'image_bytes' dataset of dtype h5py.vlen_dtype(np.dtype(np.uint8))
        - 'captions' dataset of dtype h5py.string_dtype(encoding='utf-8')
        - 'filenames' dataset of dtype h5py.string_dtype(encoding='utf-8')
        - 'drive_sequence_id' dataset of dtype h5py.string_dtype(encoding='utf-8')

    

    Usage:
    python generate_h5dataset_from_tarfile.py \
        --tar_path=/path/to/file.tar \
        --caption_root=/path/to/caption_dir/ \
        --output_h5_root=/path/to/h5_files/ \
        --output_img_size=512


    """
    fire.Fire(CommandLineConfig)
