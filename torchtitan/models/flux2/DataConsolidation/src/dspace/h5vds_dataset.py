import io
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import center_crop


class H5VDSDataset(Dataset):
    def __init__(
        self,
        vds_path: str | Path,
        target_size: Tuple[int, int] = (512, 512),
        out_dtype=torch.float32,
        augment_camera: bool = False,
        augment_camera_prob: Optional[float] = 0.5,
        camera_params_path: Optional[str | Path] = None,
    ):
        if isinstance(vds_path, str):
            vds_path = Path(vds_path)
        self.vds_path = vds_path

        # Determine length from /image_bytes (all ds are aligned along the keys)
        with h5py.File(self.vds_path, "r") as hdf:
            self._len = len(hdf["image_bytes"])

        self._file = None
        self._image_bytes = None
        self._captions = None
        self._filenames = None

        # Transforms
        self.target_size = target_size
        self.out_dtype = out_dtype
        self.transforms = v2.Compose(
            [
                v2.Resize(target_size),
                v2.PILToTensor(),
                v2.ToDtype(out_dtype, scale=True),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # Camera augmentation setup
        self.augment_camera = augment_camera
        self.augment_camera_prob = augment_camera_prob
        if isinstance(camera_params_path, str):
            camera_params_path = camera_params_path
        self.camera_params_path = camera_params_path

        if self.augment_camera:
            assert (
                self.camera_params_path is not None
            ), "camera_params_path must e provided when augment_camera=True"
            with open(self.camera_params_path, "r") as fh:
                self.intrinsic_calibrations: Dict[str, List] = yaml.load(
                    fh, Loader=yaml.SafeLoader
                )
            self.cameras = list(self.intrinsic_calibrations.keys())

    def __len__(self) -> int:
        return self._len

    def _lazy_open(self):
        if self._file is None:
            # Open *inside* each worker process
            self._file = h5py.File(self.vds_path, "r")
            self._image_bytes = self._file["image_bytes"]
            self._captions = self._file["captions"]
            self._filenames = self._file["filenames"]

    def __getitem__(self, index: int):
        self._lazy_open()

        img_blob = self._image_bytes[index]
        caption = self._captions[index]
        filename = self._filenames[index]
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8")

        with io.BytesIO(img_blob) as byte_stream:
            with Image.open(byte_stream) as img:
                img = img.convert("RGB")
                orig_w, orig_h = img.size

                # Optional random center-crop that effects FOVs
                if self.augment_camera and self.augment_camera_prob is not None:
                    if random.random() > self.augment_camera_prob:
                        new_h = random.randint(orig_h // 2, orig_h)
                        new_w = random.randint(orig_w // 2, orig_w)
                        img = center_crop(img, output_size=[new_h, new_w])
                        hfov, vfov = self._get_focal_lengths(filename, new_w, new_h)
                        aspect = float(new_w) / float(new_h)
                    else:
                        hfov, vfov = self._get_focal_lengths(filename, orig_w, orig_h)
                        aspect = float(orig_w) / float(orig_h)
                else:
                    hfov, vfov = -1.0, -1.0
                    aspect = -1.0

                img_t = self.transforms(img)

        return {
            "jpg": img_t,
            "txt": caption,
            "fovs": torch.tensor([hfov, vfov], dtype=torch.float32),
            "aspect": torch.tensor(aspect, dtype=torch.float32),
        }

    def _get_focal_lengths(
        self, filename: bytes | str, width: int, height: int
    ) -> Tuple[float, float]:
        fname = (
            filename.decode("utf-8")
            if isinstance(filename, (bytes, bytearray))
            else str(filename)
        )
        for camera in self.cameras:
            if camera in fname:
                K = np.array(self.intrinsic_calibrations[camera])
                fx, fy = K[0][0], K[1][1]
                hfov = 2 * np.arctan(width / (2.0 * fx)) * 180.0 / np.pi
                vfov = 2 * np.arctan(height / (2.0 * fy)) * 180.0 / np.pi
                return float(hfov), float(vfov)
        return 0.0, 0.0

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass


if __name__ == "__main__":
    ds = H5VDSDataset(
        vds_path="/p/data1/nxtaim/proprietary/dspace/train/train_merged_vds.h5"
    )

    dl = DataLoader(ds, batch_size=20, shuffle=True, num_workers=16)

    for idx, item in enumerate(dl):
        print(idx)
        break
