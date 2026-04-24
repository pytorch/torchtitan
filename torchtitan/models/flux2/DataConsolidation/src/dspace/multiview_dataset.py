import io
from pathlib import Path
import torch
from torch.utils.data import Dataset
import h5py
from PIL import Image
from typing import Callable, Optional, List, Set, Tuple
from torchvision.transforms.v2 import Compose
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2.functional import InterpolationMode


class MultiviewDataset(Dataset):
    def __init__(
        self,
        h5_filename: str | Path,
        input_size: int,
        transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        use_InternVL3: bool = False,
    ):
        self.h5_filename = h5_filename

        with h5py.File(self.h5_filename, "r") as hdf:
            self._len = len(hdf["front_png"])

        self.h5_file = None
        self.input_size = input_size
        self.transforms = transforms
        self.use_InternVL3 = use_InternVL3

        if self.transforms is None and self.use_InternVL3:
            self.transforms = self._build_transform(self.input_size)

    def __len__(self):
        return self._len

    def _build_transform(
        self, input_size: int
    ) -> Callable[[Image.Image], torch.Tensor]:
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet values
        transform = v2.Compose(
            [
                v2.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                v2.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                v2.ToTensor(),
                v2.Normalize(mean=MEAN, std=STD),
            ]
        )

        return transform

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image(self, image: Image.Image, input_size: int = 448, max_num: int = 12):
        transform = self._build_transform(input_size)
        images = self._dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True
        )
        pixel_values = [self.transforms(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filename, "r")

        front_img_str = self.h5_file["front_png"][index]
        rear_img_str = self.h5_file["rear_png"][index]
        front_id = self.h5_file["front_id"][index].decode("utf-8")
        rear_id = self.h5_file["rear_id"][index].decode("utf-8")
        front_caption = self.h5_file["front_caption"][index].decode("utf-8")
        rear_caption = self.h5_file["rear_caption"][index].decode("utf-8")

        with io.BytesIO(front_img_str) as bytestream:
            with Image.open(bytestream) as img:
                front_img = img.convert("RGB")
                if self.use_InternVL3:
                    front_img = self._load_image(front_img)

        with io.BytesIO(rear_img_str) as bytestream:
            with Image.open(bytestream) as img:
                rear_img = img.convert("RGB")
                if self.use_InternVL3:
                    rear_img = self._load_image(rear_img)

        return {
            "front_img": front_img,
            "rear_img": rear_img,
            "front_id": front_id,
            "rear_id": rear_id,
            "front_caption": front_caption,
            "rear_caption": rear_caption,
        }


if __name__ == "__main__":
    h5_filename = Path("/local/svenbur/nxt_aim/multiview_dataset/20250114_145353_00.h5")

    ds = MultiviewDataset(
        h5_filename, input_size=448, transforms=None, use_InternVL3=False
    )

    for idx, item in enumerate(ds):
        if idx % 2500 == 0:
            print(idx)
            item["front_img"].save(f"_{idx:04d}_front.png")
            print(item["front_caption"])
            item["rear_img"].save(f"_{idx:04d}_rear.png")
            print(item["rear_caption"])
            print("=====================================")
