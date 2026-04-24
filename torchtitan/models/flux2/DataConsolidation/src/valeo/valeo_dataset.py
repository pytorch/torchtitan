#MODIFIED VERSION 

# Same old dataloader just modified the class ValeoDatasetPreprocessed to read from metadata_5fps.parquet and merged_cameras_5fps.mp4
# and adjusted it to read the new structure of metadata_5fps.parquet file. 
# Pulls merged video frame, slices single frame, then performs normalizing and resizing and then loads 1 view, iterates again for 6 views then moves to next merged frame.



# taken from https://github.com/nxtAIM/valeo_dataloader/blob/master/valeo_dataset.py
# on 2025-11-12 14:35

import os
import json
from dataclasses import dataclass
from enum import Enum
from functools import partial
from glob import glob
from math import pi
from typing import Literal, Optional, TypedDict
from pathlib import Path

import av
import cv2
import orjson
import polars as pl
import pyarrow.parquet as pq
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import numpy as np

KNOWN_SENSOR_IDS = {
    "HDL64E_ID_4",
    "VLP32C_ID_30",
    "VLP32C_ID_31",
    "IMAR",
    "CAM_ID_70",
    "CAM_ID_71",
    "CAM_ID_72",
    "CAM_ID_73",
    "CAM_ID_74",
    "CAM_ID_75",
    "SCALA_ID_62",
    "SCALA_ID_63",
    "SCALA_ID_65",
    "SCALA_ID_66",
    "SCALA_ID_67",
    "SCALA_ID_69",
}


class SensorType(Enum):
    CAMERA = "camera"
    SCALA = "scala"
    VELODYNE = "velodyne"
    GNSS = "gnss"


class SensorFileItem(TypedDict):
    data: str
    timestamps: Optional[str]
    sensor_type: SensorType
    sensor_id: str


def extract_frame_via_av(
    video_path,
    frame_idx,
    order: Literal["HWC", "CHW"] = "CHW",
    colorspace: Literal["BGR", "RGB"] = "BGR",
) -> torch.Tensor:
    """
    Extracts a frame from a video using pyav making use of efficient jump-ahead
    to the closest keyframe.

    Args:
        video_path: Path to the video file
        frame_idx: Zero-indexed frame number to read
        order (Literal["HWC", "CHW"], optional): The channel order of the output tensor. Defaults to "CHW".
        colorspace (Literal["BGR", "RGB"], optional): The colorspace of the output tensor. Defaults to "BGR".

    Returns:
        Frame as a numpy array (or None if frame_id is out of range)
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Seek to approximate position (not exact)
    if frame_idx > 0:
        time_base = stream.time_base
        pts = int(frame_idx / stream.average_rate * (1 / time_base))
        container.seek(pts, stream=stream)

    arr = None
    target_pts = frame_idx / stream.average_rate / time_base
    # Iterate through frames to find the exact one
    for frame in container.decode(video=0):
        if frame.pts >= target_pts:
            arr = frame.to_ndarray(format="rgb24")
            break

    container.close()
    if arr is None:
        raise ValueError(f"Frame {frame_idx} not found in {video_path}")

    if colorspace == "BGR":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    tensor = torch.from_numpy(arr)

    if order.lower() == "chw":
        tensor = tensor.permute(2, 0, 1)

    return tensor


def extract_frame_via_cv2(
    video_path,
    frame_idx,
    order: Literal["HWC", "CHW"] = "CHW",
    colorspace: Literal["BGR", "RGB"] = "RGB",
) -> torch.Tensor:
    """
    Extracts a frame from a video using cv2 making use of efficient jump-ahead
    to the closest keyframe.

    Args:
        video_path (str): Path to the video file
        frame_idx (int): Zero-indexed frame number to read
        order (Literal["HWC", "CHW"], optional): The channel order of the output tensor. Defaults to "CHW".
        colorspace (Literal["BGR", "RGB"], optional): The colorspace of the output tensor. Defaults to "BGR".


    Returns:
        torch.Tensor: The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    if order.lower() not in ["hwc", "chw"]:
        raise ValueError(f"Unkown order '{order}' is supposed to be 'CHW' or 'HWC'")

    if colorspace.lower() not in ["bgr", "rgb"]:
        raise ValueError(
            f"Unkown colorspace '{colorspace}' is supposed to be 'BGR' or 'RGB'"
        )

    # lock needs write access to the directory as well
    #lock = FileLock(video_path.replace(".mp4", ".lock"))
    #with lock:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        if cap is None:
            cap_reason = "video capture is None"
        else:
            cap_reason = f"has {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames"
        raise ValueError(
            f"Could not read frame {frame_idx} from {video_path}\n"
            + f"{cap_reason}"
        )
    cap.release()

    if colorspace == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame).float()/127.5 - 1

    if order.lower() == "chw":
        tensor = tensor.permute(2, 0, 1)

    return tensor


def extract_frame_parquet(
    path, frame_idx, read_columns: Optional[list[str]] = None
) -> torch.Tensor: # Return type is actually a tensor
    """
    Extracts a row group from a parquet file and converts it to a torch tensor.
    """
    reader = pq.ParquetFile(path)
    table = pl.from_arrow(reader.read_row_group(frame_idx))
    if read_columns is not None:
        table = table.select(read_columns)
    # The original file was missing .to_torch() here. Let's ensure it's present.
    return table.to_torch()

def padded_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_shape = max(t.shape for t in tensors)
    padded_tensors = [
        torch.nn.functional.pad(t, (0, 0, 0, max_shape[0] - t.shape[0]), "constant", 0)
        for t in tensors
    ]
    return torch.stack(padded_tensors, dim=0)

def padded_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_shape = max(t.shape for t in tensors)
    padded_tensors = [
        torch.nn.functional.pad(t, (0, 0, 0, max_shape[0] - t.shape[0]), "constant", 0)
        for t in tensors
    ]


@dataclass
class DatasetConfig:
    reference_sensor_id: str
    base_path: Optional[str] = None
    camera_dirname: Optional[str] = "camera"
    scala_dirname: Optional[str] = "scala"
    velodyne_dirname: Optional[str] = "velodyne"
    gnss_dirname: Optional[str] = "IMAR"

    camera_path: Optional[str] = None
    scala_path: Optional[str] = None
    velodyne_path: Optional[str] = None
    gnss_path: Optional[str] = None

    timestamp_csv_path: Optional[str] = None
    max_timestamp_diff_seconds: float = 10e-6

    load_sensors: tuple[SensorType] = (
        SensorType.CAMERA,
        SensorType.SCALA,
        SensorType.VELODYNE,
        SensorType.GNSS,
    )
    load_sensor_ids: Optional[list[str] | set[str]] = None
    legacy_camera_filename: bool = True
    save_gnss_parquet: bool = True
    read_columns_scala: Optional[list[str]] = ("x", "y", "z", "intensity")
    read_columns_velodyne: Optional[list[str]] = ("x", "y", "z", "intensity")

    def __post_init__(self):
        # Check if either base_path or individual _path vars are set
        if self.base_path is None and not all(
            [self.camera_path, self.scala_path, self.velodyne_path, self.gnss_path]
        ):
            raise ValueError(
                "Either 'base_path' must be provided, or all individual paths "
                "('camera_path', 'scala_path', 'velodyne_path', 'gnss_path') "
                "must be set."
            )

        if self.base_path is not None:
            if self.camera_path is None:
                self.camera_path = os.path.join(self.base_path, self.camera_dirname)
            if self.scala_path is None:
                self.scala_path = os.path.join(self.base_path, self.scala_dirname)
            if self.velodyne_path is None:
                self.velodyne_path = os.path.join(self.base_path, self.velodyne_dirname)
            if self.gnss_path is None:
                self.gnss_path = os.path.join(self.base_path, self.gnss_dirname)

        if self.load_sensor_ids is None:
            self.load_sensor_ids = KNOWN_SENSOR_IDS


class ValeoDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        super().__init__()
        #print("Init dataset with config:")
        #for attr in dir(config):
        #    if not attr.startswith("_"):
        #        print(f" - {attr:>22}: {getattr(config, attr, "COULD NOT ACCESS")}")
        #print()
        self.config = config
        self.load_sensor_ids = (
            set(config.load_sensor_ids) if config.load_sensor_ids else None
        )

        self.load_sensors = self._get_true_load_sensors(
            config.load_sensors, self.config.load_sensor_ids
        )

        self.data_paths_dict: dict[str, SensorFileItem] = {}
        for sensor_type in self.load_sensors:
            path = getattr(self.config, f"{sensor_type.value}_path")
            if path is None:
                continue
            data_paths = self._find_data_paths(path, sensor_type)
            self.data_paths_dict.update(
                {
                    i["sensor_id"]: i
                    for i in data_paths
                    if i["sensor_id"] in self.load_sensor_ids
                }
            )

        ref_found = False
        for item in self.data_paths_dict.values():
            if item["sensor_id"] == self.config.reference_sensor_id:
                ref_found = True
                break
        if not ref_found:
            raise ValueError(
                f"Reference sensor data '{self.config.reference_sensor_id}' not found in the specified directory."
            )
        self.gnss = None
        if SensorType.GNSS in self.load_sensors:
            self.gnss = self._load_gnss(self.data_paths_dict["IMAR"]["data"])
            if self.config.save_gnss_parquet:
                self.gnss.write_parquet(
                    self.data_paths_dict["IMAR"]["data"].replace(".json", ".parquet")
                )

        self.timestamps_df = self._collate_timestamps(
            self.data_paths_dict.values(), self.gnss
        )

        self.matches = self._find_matches(
            self.timestamps_df,
            self.config.reference_sensor_id,
            self.config.max_timestamp_diff_seconds,
        )
        if len(self.matches) == 0:
            raise ValueError(
                "No matches found within the specified timestamp difference."
            )

        self.loaders = self._make_loaders(self.data_paths_dict)

    def _get_true_load_sensors(
        self, load_sensors: list[SensorType], load_sensor_ids: Optional[set[str]] = None
    ):
        if load_sensor_ids is None:
            return load_sensors
        true_load_sensors = []
        for sensor_id in load_sensor_ids:
            match sensor_id:
                case "IMAR":
                    true_load_sensors.append(SensorType.GNSS)
                case "HDL64E_ID_4" | "VLP32C_ID_30" | "VLP32C_ID_31":
                    true_load_sensors.append(SensorType.VELODYNE)
                case str() if sensor_id.startswith("CAM_ID_"):
                    true_load_sensors.append(SensorType.CAMERA)
                case str() if sensor_id.startswith("SCALA_ID_"):
                    true_load_sensors.append(SensorType.SCALA)
                case _:
                    raise ValueError(f"Unknown sensor id: {sensor_id}")

        return true_load_sensors

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        out = {}
        for sensor_type in self.loaders:
            out_per_sensor = []
            sorted_per_id = sorted(
                [(s_id, fn) for s_id, fn in self.loaders[sensor_type].items()],
                key=lambda x: x[0],
            )

            for sensor_id, load_fn in sorted_per_id:
                frame_id = self._get_frame_id(sensor_id, self.matches[idx])
                out_per_sensor.append(load_fn(frame_idx=frame_id))

            if sensor_type in (SensorType.SCALA, SensorType.VELODYNE):
                out[sensor_type.value] = padded_stack(out_per_sensor)
            else:
                out[sensor_type.value] = torch.stack(out_per_sensor, dim=0)

        return out

    def _cam_id_from_path(self, path: str, legacy: bool = False) -> str:
        """
        Parses the camera ID from a given path.

        Args:
            path (str): The path to the camera data (e.g., video file).
            legacy (bool): If True, uses a parsing method from legacy camera filenames.

        Returns:
            str: The extracted camera ID.
        """
        base = os.path.basename(path).replace(".mp4", "")
        if not legacy:
            return base
        return f"CAM_ID_{base.split('_')[-1]}"

    def _get_frame_id(self, sensor_id: str, matches: pl.DataFrame):
        """
        Extracts the frame id for a given sensor id from a dataframe of matches.

        Args:
            sensor_id (str): The sensor id for which to extract the frame id.
            matches (pl.DataFrame): The dataframe containing the matched sensor data.

        Returns:
            int: The extracted frame id.
        """

        return matches.filter(pl.col("sensor_id") == sensor_id)["frame_id"].item()

    def _make_loaders(self, data_paths: dict[str, dict]):
        """
        Creates a dictionary of loader functions for each sensor type and sensor ID.

        Args:
            data_paths (dict): A dictionary mapping sensor IDs to their respective data.

        Returns:
            dict: A dictionary where keys are `SensorType` enums and values are
                  dictionaries mapping sensor IDs to their respective loader functions.
        """
        loaders = {}
        for sensor_id, item in data_paths.items():
            if item["sensor_type"] not in loaders:
                loaders[item["sensor_type"]] = {}
            match item["sensor_type"]:
                case SensorType.CAMERA:
                    loaders[SensorType.CAMERA][sensor_id] = partial(
                        extract_frame_via_cv2, video_path=item["data"], order="CHW"
                    )
                case SensorType.SCALA:
                    loaders[item["sensor_type"]][sensor_id] = partial(
                        extract_frame_parquet,
                        path=item["data"],
                        read_columns=self.config.read_columns_scala,
                    )
                case SensorType.VELODYNE:
                    loaders[item["sensor_type"]][sensor_id] = partial(
                        extract_frame_parquet,
                        path=item["data"],
                        read_columns=self.config.read_columns_velodyne,
                    )
                case SensorType.GNSS:
                    loaders[item["sensor_type"]][sensor_id] = (
                        lambda frame_idx: self.gnss.filter(
                            pl.col("frame_id") == frame_idx
                        )["lat", "lon", "host", "device"].to_torch()
                    )
        return loaders

    def _find_data_paths(
        self, path: str, sensor_type: SensorType
    ) -> list[SensorFileItem]:
        """
        Finds data and timestamp paths for a given sensor type in a directory.

        Args:
            path (str): The path to the directory containing the data.
            sensor_type (SensorType): The type of sensor data to find.

        Returns:
            list[SensorFileItem]: A list of dictionaries containing the path and auxillary data
        """
        paths = []
        match sensor_type:
            case SensorType.CAMERA:
                files = sorted(list(glob(os.path.join(path, "*.mp4"))))
                timestamps = [f.replace(".mp4", "_timestamps.csv") for f in files]
                paths = [
                    SensorFileItem(
                        data=v,
                        timestamps=t,
                        sensor_type=sensor_type,
                        sensor_id=self._cam_id_from_path(
                            v, legacy=self.config.legacy_camera_filename
                        ),
                    )
                    for v, t in zip(files, timestamps)
                ]
            case SensorType.SCALA | SensorType.VELODYNE:
                files = sorted(list(glob(os.path.join(path, "*.parquet"))))
                timestamps = [f.replace(".parquet", "_timestamps.csv") for f in files]
                paths = [
                    SensorFileItem(
                        data=v,
                        timestamps=t,
                        sensor_type=sensor_type,
                        sensor_id=os.path.basename(v).replace(".parquet", ""),
                    )
                    for v, t in zip(files, timestamps)
                ]
            case SensorType.GNSS:
                files = list(glob(os.path.join(path, "IMAR.parquet")))
                if len(files) == 0:
                    files = list(glob(os.path.join(path, "IMAR.json")))
                paths = [
                    SensorFileItem(
                        data=f,
                        timestamps=None,
                        sensor_type=sensor_type,
                        sensor_id="IMAR",
                    )
                    for f in files
                ]
            case _:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

        return paths

    def _load_gnss(self, path: str) -> pl.DataFrame:
        """
        Loads GNSS data from a JSON or Parquet file.

        Args:
            path (str): The path to the GNSS data file.

        Returns:
            pl.DataFrame: The loaded GNSS data as a Polars DataFrame with columns ['lat', 'lon', 'host', 'device']. \\
                          The `host` and `device` columns contain the timestamps.
        """
        if path.endswith(".parquet"):
            return pl.read_parquet(path)

        with open(path, "rb") as f:
            data = orjson.loads(f.read())

        f_data = []
        pos = {}
        frame_id = 0

        for host, device, item in zip(
            data["sw_timestamps"], data["hw_timestamps"], data["msgs"]
        ):
            if item["msg_name"] == "GNSS_Latitude":
                pos["lat"] = (item["Latitude"] / pi) * 180
                pos["host"] = host
                pos["device"] = device
            if item["msg_name"] == "GNSS_Longitude":
                pos["lon"] = (item["Longitude"] / pi) * 180
                pos["host"] = host
                pos["device"] = device
            if "lat" in pos and "lon" in pos:
                pos["frame_id"] = frame_id
                frame_id += 1
                f_data.append(pos)
                pos = {}

        df = pl.from_dicts(f_data)
        return df

    def _collate_timestamps(
        self, data_paths: list[SensorFileItem], gnss: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Combines timestamps from found sensors into a single DataFrame.

        Args:
            data_paths (list[SensorFileItem]): A list of dictionaries containing the path and auxillary data
            gnss (Optional[pl.DataFrame]): Optional GNSS data. Avoids double loading if passed.

        Returns:
            pl.DataFrame: A Polars DataFrame containing collated timestamps from all sensors.
        """
        dfs = []
        for item in data_paths:
            match item["sensor_type"]:
                case SensorType.CAMERA:
                    timestamp_df = pl.read_csv(item["timestamps"])
                    timestamp_df_formatted = pl.DataFrame(
                        {
                            "frame_id": timestamp_df["frame_id"].cast(pl.Int64),
                            "host": timestamp_df["sw_timestamp"].cast(pl.Float64),
                            "device": timestamp_df["ptp_timestamp"].cast(pl.Float64),
                            "sensor_id": item["sensor_id"],
                            "sensor_type": item["sensor_type"].value,
                        }
                    )
                    dfs.append(timestamp_df_formatted)
                case SensorType.SCALA | SensorType.VELODYNE:
                    timestamp_df = pl.read_csv(item["timestamps"])
                    timestamp_df_formatted = pl.DataFrame(
                        {
                            "frame_id": list(range(len(timestamp_df))),
                            "host": timestamp_df["host"].cast(pl.Float64),
                            "device": timestamp_df["device"].cast(pl.Float64),
                            "sensor_id": item["sensor_id"],
                            "sensor_type": item["sensor_type"].value,
                        }
                    )
                    dfs.append(timestamp_df_formatted)
                case SensorType.GNSS:
                    if gnss is not None:
                        df = gnss
                    else:
                        df = self._load_gnss(item["data"])
                    timestamp_df_formatted = pl.DataFrame(
                        {
                            "frame_id": df["frame_id"].cast(pl.Int64),
                            "host": df["host"].cast(pl.Float64),
                            "device": df["device"].cast(pl.Float64),
                            "sensor_id": item["sensor_id"],
                            "sensor_type": item["sensor_type"].value,
                        }
                    )
                    dfs.append(timestamp_df_formatted)

                    pass
                case _:
                    raise ValueError(f"Unknown sensor type: {item['sensor_type']}")

        df = pl.concat(dfs)
        df = df.sort("host")
        return df

    def _find_matches(
        self,
        timestamp_df: pl.DataFrame,
        reference_sensor_id: str,
        max_timestamp_diff: float,
    ) -> list[pl.DataFrame]:
        """
        For each row with reference_sensor_id, find the closest row from each other sensor_id
        within the time threshold. Optimized for speed.

        Args:
            timestamp_df (pl.DataFrame): A Polars DataFrame containing timestamps.
            reference_sensor_id (str): The ID of the reference sensor.
            max_timestamp_diff (float): Maximum difference of timestamps in seconds counted as a match.

        Returns:
            list[pl.DataFrame]: A list of Polars DataFrames, where each DataFrame represents a matched set of sensor data for a given reference frame.
        """
        reference_rows = timestamp_df.filter(pl.col("sensor_id") == reference_sensor_id)
        other_rows = timestamp_df.filter(pl.col("sensor_id") != reference_sensor_id)

        if len(other_rows) == 0:
            return reference_rows.with_columns(time_diff=0.0).partition_by("frame_id")

        sensor_ids = other_rows["sensor_id"].unique().sort().to_list()
        num_sensors = len(sensor_ids)
        tolerance_us = int(max_timestamp_diff * 1e6)

        # Join each sensor to reference frames
        sensor_matches = []
        for sensor_id in sensor_ids:
            sensor_data = other_rows.filter(pl.col("sensor_id") == sensor_id)

            matched = (
                reference_rows.join_asof(
                    sensor_data,
                    on="host",
                    strategy="nearest",
                    coalesce=False,
                    tolerance=tolerance_us,
                )
                .filter(pl.col("frame_id_right").is_not_null())
                .select(
                    [
                        pl.col("frame_id").alias("ref_frame_id"),
                        pl.col("frame_id_right").alias("frame_id"),
                        pl.col("sensor_id_right").alias("sensor_id"),
                        pl.col("host_right").alias("host"),
                        (pl.col("host") - pl.col("host_right"))
                        .abs()
                        .alias("time_diff"),
                    ]
                )
            )
            sensor_matches.append(matched)

        # Stack all matches
        all_matches = pl.concat(sensor_matches)

        # Count matches per reference frame
        frame_counts = (
            all_matches.group_by("ref_frame_id")
            .agg(pl.col("sensor_id").n_unique().alias("sensor_count"))
            .filter(pl.col("sensor_count") == num_sensors)
        )

        if len(frame_counts) == 0:
            return []

        complete_matches = all_matches.join(
            frame_counts.select("ref_frame_id"),
            on="ref_frame_id",
            how="semi",
        )

        # Add reference rows
        ref_rows = reference_rows.join(
            frame_counts.select("ref_frame_id"),
            left_on="frame_id",
            right_on="ref_frame_id",
            how="semi",
        ).select(
            [
                pl.col("frame_id").alias("ref_frame_id"),
                pl.col("frame_id"),
                pl.col("sensor_id"),
                pl.col("host"),
                pl.lit(0.0).alias("time_diff"),
            ]
        )

        # Combine and partition
        final_df = pl.concat([complete_matches, ref_rows]).sort("host")

        return final_df.partition_by("ref_frame_id", maintain_order=True, as_dict=False)



class ValeoDatasetCameraMerged(Dataset):
    """
    A fast dataloader that reads from a preprocessed Valeo dataset directory.
    This version is optimized for use with a multi-worker torch.utils.data.DataLoader.
    """
    def __init__(self, config: DatasetConfig, camera_grid_shape: tuple[int, int] = (2, 3)):
        super().__init__()
        self.config = config
        self.base_path = Path(config.base_path)
        self.camera_grid_shape = camera_grid_shape
        
        metadata_path = self.base_path / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}\n"
                                    "Please run the preprocessing script first.")
        
        all_matches = pl.read_parquet(metadata_path)
        self.matches = all_matches.partition_by("ref_frame_id", maintain_order=True, as_dict=False)
        
        self.sensor_ids = all_matches["sensor_id"].unique().to_list()
        self.sensor_types = {
            sid: stype for sid, stype in all_matches.select("sensor_id", "sensor_type").unique().iter_rows()
        }

        self.video_path = str(self.base_path / self.config.camera_dirname / "merged_cameras.mp4")
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Merged video not found: {self.video_path}")
            
        self.cam_ids = sorted([sid for sid in self.sensor_ids if sid.startswith("CAM_ID_")])
        
        # This will be initialized to None in each worker process.
        self.cap = None
        
        # Pre-load GNSS data in the main process
        self.gnss_df = None
        if 'IMAR' in self.sensor_ids:
            gnss_path = self.base_path / self.config.gnss_dirname / "IMAR.parquet"
            if gnss_path.exists():
                self.gnss_df = pl.read_parquet(gnss_path)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
         # --- THIS IS THE CRITICAL FIX ---
        # Each worker process gets its own 'self.cap'.
        # This check ensures the VideoCapture object is created ONCE PER WORKER.
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                # This error will now correctly point to the failing worker.
                raise IOError(f"Could not open video file for worker: {self.video_path}")

        final_out = {}
        match_group = self.matches[idx]
        
        # 1. Load and process camera data
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, grid_frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {idx} from {self.video_path}.")

        grid_frame_rgb = cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB)
        grid_tensor = (torch.from_numpy(grid_frame_rgb).float() / 127.5) - 1.0

        h, w = grid_tensor.shape[0] // self.camera_grid_shape[0], grid_tensor.shape[1] // self.camera_grid_shape[1]
        
        cam_tensors = [
            grid_tensor[
                (i // self.camera_grid_shape[1]) * h : ((i // self.camera_grid_shape[1]) + 1) * h,
                (i % self.camera_grid_shape[1]) * w : ((i % self.camera_grid_shape[1]) + 1) * w,
                :
            ].permute(2, 0, 1)
            for i in range(len(self.cam_ids))
        ]
        
        if cam_tensors:
            final_out['jpg'] = cam_tensors[0]
        final_out['txt'] = "A photorealistic image from a front-facing car camera."

        # 2. Load other sensor data (optional, for debugging or other uses)
        scala_data, velodyne_data, gnss_data = [], [], []
        
        for sensor_id in self.sensor_ids:
            sensor_type_str = self.sensor_types.get(sensor_id)
            if sensor_type_str == 'camera':
                continue

            frame_id_series = match_group.filter(pl.col("sensor_id") == sensor_id)["frame_id"]
            if frame_id_series.is_empty():
                continue
            frame_id = frame_id_series.item()

            if sensor_type_str in ('scala', 'velodyne'):
                dirname = getattr(self.config, f"{sensor_type_str}_dirname")
                path = self.base_path / dirname / f"{sensor_id}.parquet"
                read_cols = getattr(self.config, f"read_columns_{sensor_type_str}")
                
                data = extract_frame_parquet(str(path), frame_id, read_columns=read_cols)
                
                if sensor_type_str == 'scala':
                    scala_data.append(data)
                else:
                    velodyne_data.append(data)
            
            elif sensor_type_str == 'gnss' and self.gnss_df is not None:
                gnss_row = self.gnss_df.filter(pl.col("frame_id") == frame_id)
                if not gnss_row.is_empty():
                    gnss_data.append(gnss_row.to_torch())

        # Add optional data to the output dictionary
        if scala_data:
            final_out['scala'] = padded_stack(scala_data)
        if velodyne_data:
            final_out['velodyne'] = padded_stack(velodyne_data)
        if gnss_data:
            final_out['gnss'] = torch.cat(gnss_data, dim=0) if gnss_data else torch.empty(0)
            
        return final_out

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()

class ValeoDatasetPreprocessed(Dataset):
    def __init__(self, config, camera_grid_shape: tuple[int, int] = (2, 3)):
        super().__init__()
        self.config = config
        self.base_path = Path(config.base_path)
        self.rows, self.cols = camera_grid_shape
        
        self.item_index = []
        self.rec_registry = {} 
        self.caption_cache = {} # Stores captions for the CURRENT recording only

        print(f"Deep scanning {self.base_path} for 10FPS preprocessed recordings...")
        metadata_files = list(self.base_path.rglob("metadata_10fps.parquet"))
        
        for metadata_path in tqdm(metadata_files, desc="Indexing Dataset"):
            rec_root = metadata_path.parent
            video_path = rec_root / "camera" / "merged_cameras_10fps.mp4"
            caption_path = rec_root / "captions.json" # Link to your new JSON

            if not video_path.exists(): continue

            try:
                df = pl.read_parquet(metadata_path)
                rec_name = rec_root.name
                
                # Check video metadata
                cap_check = cv2.VideoCapture(str(video_path))
                actual_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_check.release() 
                
                if actual_frames <= 0: continue

                # Store paths in registry
                self.rec_registry[rec_name] = {
                    "video_path": str(video_path),
                    "caption_path": str(caption_path) if caption_path.exists() else None
                }
                
                valid_frame_count = min(len(df), actual_frames)
                for i in range(valid_frame_count):
                    self.item_index.append({'rec_id': rec_name, 'frame_idx': i})
            except Exception:
                continue

    def __len__(self):
        return len(self.item_index) * 6

    def __getitem__(self, idx):
        video_frame_idx = idx // 6
        camera_slice_idx = idx % 6

        item_info = self.item_index[video_frame_idx]
        rec_id = item_info['rec_id']
        rec_data = self.rec_registry[rec_id]

        # --- CAPTION LOADING LOGIC ---
        caption = "A photorealistic car camera image." # Default fallback
        if rec_data["caption_path"]:
            # Efficiently cache only the current recording's captions
            if rec_id not in self.caption_cache:
                # Clear cache if it gets too big (simple RAM management)
                if len(self.caption_cache) > 2: self.caption_cache.clear()
                
                with open(rec_data["caption_path"], 'r') as f:
                    self.caption_cache[rec_id] = json.load(f)
            
            # Construct the EXACT key your captioner script created
            # Format: rec_name_frame_X_view_Y
            caption_key = f"{rec_id}_frame_{item_info['frame_idx']}_view_{camera_slice_idx}"
            #caption = self.caption_cache[rec_id].get(caption_key, caption)
            if caption_key in self.caption_cache[rec_id]:
                caption = self.caption_cache[rec_id][caption_key]
            else:
                print(f"[DEBUG GET] ERROR: Key {caption_key} not found in JSON!")

        # --- IMAGE LOADING LOGIC ---
        cap = cv2.VideoCapture(rec_data['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item_info['frame_idx'])
        ret, grid_frame = cap.read()
        cap.release() 

        if not ret or grid_frame is None:
            return self.__getitem__((idx + 6) % len(self))

        # Slicing logic (Top-Left, Top-Mid, etc.)
        gh, gw = grid_frame.shape[0], grid_frame.shape[1]
        h, w = gh // self.rows, gw // self.cols
        row, col = camera_slice_idx // self.cols, camera_slice_idx % self.cols
        view = grid_frame[row*h : (row+1)*h, col*w : (col+1)*w, :]
        
        view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(view_rgb).permute(2, 0, 1).float()
        tensor = tensor.div_(127.5).sub_(1.0)

        if tensor.shape[-2:] != (512, 512):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            'jpg': torch.clamp(tensor, -1.0, 1.0),
            'txt': caption 
        }

def __del__(self):
        # Proper cleanup of resources
        for rec in self.rec_registry.values():
            if 'cap' in rec and rec['cap'] is not None:
                rec['cap'].release()

def main():
    import argparse
    import time

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/p/data1/nxtaim/proprietary_samples/valeo/")
    parser.add_argument("--ref", type=str, default="HDL64E_ID_4")
    parser.add_argument(
        "--load-sensors",
        nargs="+",
        type=SensorType,
        default=[
            SensorType.CAMERA,
            SensorType.SCALA,
            SensorType.VELODYNE,
            SensorType.GNSS,
        ],
    )
    args = parser.parse_args()

    config = DatasetConfig(
        base_path=args.base_path,
        reference_sensor_id="HDL64E_ID_4",
        load_sensors=args.load_sensors,
        # load_sensor_ids=["CAM_ID_70", "IMAR"],
        legacy_camera_filename=True,
        save_gnss_parquet=False, # needs write permissions
        max_timestamp_diff_seconds=0.03,
    )
    start = time.perf_counter()
    dataset = ValeoDataset(config)
    print(f"Init time: {time.perf_counter() - start}")

    start = time.perf_counter()
    item = dataset[0]
    print(f"Item time: {time.perf_counter() - start}")
    # exit()

    try:
        import rerun as rr
    except ImportError:
        print("Could not find rerun. Skipping visualization")
        return

    rr.init("Valeo Dataset", spawn=True)
    rr.connect_grpc()
    import matplotlib.pyplot as plt

    colormap = plt.cm.turbo

    for sample_id in tqdm(range(len(dataset))):
        item = dataset[sample_id]
        rr.set_time("frame_id", sequence=sample_id)

        for sensor_type in config.load_sensors:
            if sensor_type.value not in item:
                continue
            data = item[sensor_type.value]
            match sensor_type:
                case SensorType.CAMERA:
                    for i in range(data.shape[0]):
                        rr.log(
                            f"camera/{i}",
                            rr.Image(
                                image=item[SensorType.CAMERA.value][i]
                                .permute(1, 2, 0)
                                .numpy()
                                / 255.0,
                                color_model="BGR",
                            ),
                        )
                case SensorType.SCALA | SensorType.VELODYNE:
                    for i in range(data.shape[0]):
                        intensities = data[i, :, 3]
                        normalized_intensities = (intensities - intensities.min()) / (
                            intensities.max() - intensities.min()
                        )
                        colors = colormap(normalized_intensities)[:, :3]
                        rr.log(
                            f"{sensor_type.value}/{i}",
                            rr.Points3D(
                                positions=data[i, :, :3],
                                colors=colors,
                            ),
                        )

                case SensorType.GNSS:
                    rr.log(
                        "gnss",
                        rr.GeoPoints(
                            lat_lon=item[SensorType.GNSS.value][..., :2].reshape(1, 2)
                        ),
                    )


if __name__ == "__main__":
    main()