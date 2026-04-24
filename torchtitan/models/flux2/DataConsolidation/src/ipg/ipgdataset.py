import os, sys
import re
import cv2
import numpy as np

from pathlib import Path
from typing import Callable, Literal, Tuple, TypedDict, Optional, Union
from typing_extensions import override

import torch
from torchvision.transforms import v2

basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not basepath in sys.path:
    sys.path.insert(0, basepath)
from ..multifiledataset import MultiFileDataset


class CameraIntrinsics(TypedDict):
    width: int  # image width in pixels
    height: int  # image height in pixels
    hfov: float  # horizontal field of view in degrees
    vfov: float  # vertical field of view in degrees
    distortion: tuple[float, float, float]  # distortion coefficients ( k1, k2, k3 )


class CameraExtrinsics(TypedDict):
    position: Tuple[float, float, float]  # (x, y, z) in meters
    rotation: Tuple[float, float, float]  # (roll, pitch, yaw) in degrees


class CameraConfig(TypedDict):
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


class ScenarioVariation(TypedDict):
    time: Literal["midday", "dawn", "night"]
    weather: Literal["sunny", "overcast", "rain", "fog", "snow"]
    traffic: Literal["flowing", "congested", "uncommon"]
    camera: Literal[
        "front",
        "front_distorted",
        "left_distorted",
        "right_distorted",
        "back_distorted",
    ]


class IPGDatasetEntry(TypedDict):
    jpg: np.ndarray  # The RGB image of shape (H, W, 3) and type uint8
    variation: ScenarioVariation  # The scenario variation metadata
    fovs: Tuple[float, float]  # (hfov, vfov) in degrees
    aspect: float  # aspect ratio (width / height)
    txt: Optional[str]  # Optional text prompt


def example_prompt_fn(variation: ScenarioVariation) -> str:
    """Create an example text prompt from the variation."""

    direction = variation["camera"].split("_")[0]

    return f"A photo from a {direction}-facing camera capturing a scene during {variation['time']} with {variation['weather']} weather."


CAMERA_PINHOLE: CameraIntrinsics = {
    "width": 1920,
    "height": 1080,
    "hfov": 60.0,
    "vfov": 33.75,
    "distortion": (0.0, 0.0, 0.0),
}
CAMERA_WIDEANGLE: CameraIntrinsics = {
    "width": 1920,
    "height": 1080,
    "hfov": 120.0,
    "vfov": 67.5,
    "distortion": (0.8, -0.4, 0.1),
}

CAMERA_CONFIGS: dict[str, CameraConfig] = {
    "front": {
        "intrinsics": CAMERA_PINHOLE,
        "extrinsics": {"position": (2.8, 0.0, 1.3), "rotation": (0.0, 0.0, 0.0)},
    },
    "front_distorted": {
        "intrinsics": CAMERA_WIDEANGLE,
        "extrinsics": {"position": (2.8, 0.0, 1.4), "rotation": (0.0, 0.0, 0.0)},
    },
    "right_distorted": {
        "intrinsics": CAMERA_WIDEANGLE,
        "extrinsics": {"position": (2.5, -0.5, 1.4), "rotation": (0.0, 0.0, -45.0)},
    },
    "left_distorted": {
        "intrinsics": CAMERA_WIDEANGLE,
        "extrinsics": {"position": (2.5, 0.5, 1.4), "rotation": (0.0, 0.0, 45.0)},
    },
    "rear_distorted": {
        "intrinsics": CAMERA_WIDEANGLE,
        "extrinsics": {"position": (0.0, 0.0, 1.4), "rotation": (0.0, 0.0, 180.0)},
    },
}


class IPGDataset(MultiFileDataset):
    """
    Data loader for the provided dataset of IPG-Automotive.

    The dataset consists of video clips from various cameras mounted on a vehicle,
    capturing different weather conditions and times of day. Each clip is generated
    with the CarMaker Simulation software. The clips have lengths between 10 and 120
    seconds, a frame rate of 25 FPS, and contain both normal driving as well as edge
    case scenarios.

    Scenario variations:

    - time of day: midday, dawn, night
    - weather: sunny, overcast, rain, fog, snow
    - traffic: flowing, congested, uncommon

    Camera setup:

    - 1x pinhole camera: 1920x1080, HFOV 60°, no distortion, facing forward
    - 4x wide-angle cameras: 1920x1080, HFOV 120°, with camera distortion,
      facing forward, left, right and backward

    Output format:

    - The dataset returns a dictionary that follows the `IPGDatasetEntry` specification
    
    """

    @staticmethod
    def _variation_from_filename(filename: str) -> ScenarioVariation:
        """Create metadata from the filename."""

        match = re.fullmatch(
            r"([a-zA-Z0-9_\-]+)__mod_([a-z]+)-([a-z]+)(?:-([a-z]+))?_r0_([a-z]+)(_distorted)?\.mp4",
            filename,
        )
        if not match:
            raise ValueError(f"Filename '{filename}' does not match expected pattern.")

        scenario, daytime, weather, traffic, camera, distorted = match.groups()

        if traffic not in {"flowing", "congested"}:
            traffic = "uncommon"

        return {
            "time": daytime,
            "weather": weather,
            "traffic": traffic,
            "camera": camera + (distorted or ""),
        }  # type: ignore

    @staticmethod
    def _find_video_files(
        root: Path, filter: Optional[Callable[[ScenarioVariation], bool]]
    ) -> list[Path]:
        """Recursively find all video files in the root directory, optionally filtering by scenario variation."""

        video_files = list(root.rglob("*.mp4"))
        if not video_files:
            raise ValueError(f"No video files found in directory {root}")

        if filter is None:
            return video_files
        else:
            return [
                file
                for file in video_files
                if filter(IPGDataset._variation_from_filename(file.name))
            ]

    def __init__(
        self,
        source: str | Path,
        filter: Optional[Callable[[ScenarioVariation], bool]] = None,
        prompt: Union[None, str, Callable[[ScenarioVariation], str]] = None,
        out_img_size : tuple[int,int] | None = None,
        out_dtype: torch.dtype = torch.float32,
        transform : v2.Transform | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the IPGDataset.

        Args:
            source: Root directory that contains the dataset.
            filter: Optional function to filter video clips based on their scenario variation.
            transform: Optional function to apply transformations to each image.
            prompt: Optional function to generate a text prompt from the scenario variation.
        """

        video_files = IPGDataset._find_video_files(Path(source), filter)
        super().__init__(*args, source=video_files, **kwargs)
        self._prompt = prompt

        self.outsize = None
        self.outdtype = out_dtype
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(out_dtype),
            v2.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])  
        ]
        if transform is not None:
            transform_list.append(transform)
        if out_img_size is not None:
            assert (
                hasattr(out_img_size, "__len__") and len(out_img_size) == 2
            ), f"size needs to be a tupel, instead got: {out_img_size}"
            resize = v2.Resize((out_img_size[0], out_img_size[1]), antialias=True)
            self.outsize = out_img_size
            transform_list.append(resize)
        self._transform = v2.Compose(transform_list)

    @override
    def _get_samples_metadata_from_filepath(self, path: str):
        cap = cv2.VideoCapture(path)
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata = np.array(
            (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                float(cap.get(cv2.CAP_PROP_FPS)),
            ),
            dtype=[("height", np.uint32), ("width", np.uint32), ("fps", np.float32)],
        )
        cap.release()

        return length, metadata

    @override
    def _getitem(
        self, file_path: str, sample_idx: int, metadata: np.ndarray
    ) -> IPGDatasetEntry:
        cap = cv2.VideoCapture(file_path)
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video file {file_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {sample_idx} from {file_path}")
        cap.release()
        frame = self._transform(np.ascontiguousarray(frame[:, :, ::-1]))

        variation = IPGDataset._variation_from_filename(Path(file_path).name)
        intrinsics = CAMERA_CONFIGS[variation["camera"]]["intrinsics"]

        result = {
            "jpg": frame,
            "variation": variation,
            "fovs": (intrinsics["hfov"], intrinsics["vfov"]),
            "aspect": metadata["width"] / metadata["height"],
        }
        if self._prompt is not None:
            if isinstance(self._prompt, str):
                result["txt"] = self._prompt
            else:
                result["txt"] = self._prompt(variation)

        return result  # type: ignore

    @override
    def _get_image_representation(
        self,
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray,
        batch=None,
    ): # -> Tuple[FFMPEG_VIDEO_WRITER_ACCEPTS, str]:
        if batch is None:
            batch = self._getitem(file_path, sample_idx, metadata)
        
        img = batch["jpg"].cpu().numpy() # type: ignore # batch['jpg'] is a tensor due to ToImage
        img = np.ascontiguousarray(127.5 * img.transpose(1,2,0) + 127.5, dtype=np.uint8)

        return img, batch["txt"] if 'txt' in batch else ""