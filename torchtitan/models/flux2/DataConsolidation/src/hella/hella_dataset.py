# Taken from https://github.com/nxtAIM/hella_dataloader/blob/main/hella_dataset.py
# on 11.11.2025 21:43

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025-06-30
# @Author  : Zhaoze Wang (FORVIA HELLA)

import os
import numpy as np
from glob import glob
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader
from functools import cached_property
from typing import List, Tuple, Dict, Any
import cv2

class HellaDataset(Dataset):
    """
    Extended Hella dataset loader that supports:
      - Combining multiple sequences and sub-sequences into a single dataset.
      - Loading data selectively by sensors (e.g., only velodyne or specific cameras).
      - Flexible timestamp synchronization: uses the first sensor in `sensors` as the reference.
      - NEW: Camera images are now loaded from video.avi instead of JPG files.
    """

    def __init__(
        self,
        dataset_dir: str = "/p/data1/nxtaim/proprietary/hella/HellaDataset/",
        seq_list: List[str] = ['seq1', 'seq2'],
        sensors: List[str] = ['velodyne', 'oxts', 'cam0', 'cam6']
    ):
        self.dataset_dir = dataset_dir
        self.seq_list = seq_list
        self.sensors = sensors

        # Newly structured video containers
        self.video_caps: Dict[str, List[cv2.VideoCapture]] = {f'image_{i:02d}': [] for i in range(7)}
        self.video_frame_counts: Dict[str, List[int]] = {f'image_{i:02d}': [] for i in range(7)}

        # Timestamps storage
        self.velodyne_timestamps = []
        self.oxts_timestamps = []
        self.image_timestamps = {f'image_{i:02d}': [] for i in range(7)}

        # File path (only for velodyne and oxts)
        self.velodyne_files = []
        self.oxts_files = []

        # Load sequences
        self._load_all_sequences()

        # The first sensor is the reference
        self.reference_sensor = sensors[0]
        self._setup_reference_timestamps()

    # ----------------------------------------------------------------------
    #                         Data Loading Functions
    # ----------------------------------------------------------------------

    def _load_all_sequences(self):
        for seq_name in self.seq_list:
            for sub_seq in range(10):
                dataset_dir = os.path.join(self.dataset_dir, seq_name, str(sub_seq))

                # ------------------ Velodyne ------------------
                if 'velodyne' in self.sensors:
                    vel_dir = os.path.join(dataset_dir, 'velodyne_points', 'data')
                    self.velodyne_files += sorted(glob(os.path.join(vel_dir, '*.bin')))
                    self.velodyne_timestamps += self._load_timestamps(
                        os.path.join(dataset_dir, 'velodyne_points', 'timestamps.txt')
                    )

                # ------------------ Cameras (from video) ------------------
                for i in range(7):
                    cam_key = f'cam{i}'
                    img_key = f'image_{i:02d}'
                    if cam_key in self.sensors:
                        video_path = os.path.join(dataset_dir, f'image_{i:02d}', 'video.avi')

                        if os.path.exists(video_path):
                            cap = cv2.VideoCapture(video_path)
                            self.video_caps[img_key].append(cap)

                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            self.video_frame_counts[img_key].append(frame_count)
                        else:
                            self.video_caps[img_key].append(None)
                            self.video_frame_counts[img_key].append(0)

                        # Load timestamps
                        ts = self._load_timestamps(os.path.join(dataset_dir, f'image_{i:02d}', 'timestamps.txt'))
                        self.image_timestamps[img_key] += ts

                # ------------------ OXTS ------------------
                if 'oxts' in self.sensors:
                    self.oxts_files += sorted(glob(os.path.join(dataset_dir, 'oxts', 'data', '*.txt')))
                    self.oxts_timestamps += self._load_timestamps(
                        os.path.join(dataset_dir, 'oxts', 'timestamps.txt')
                    )

    # ----------------------------------------------------------------------

    def _setup_reference_timestamps(self):
        """
        Decide which timestamps define dataset length and synchronization.
        """
        if self.reference_sensor == 'velodyne':
            self.reference_files = self.velodyne_files
            self.reference_timestamps = self.velodyne_timestamps

        elif self.reference_sensor == 'oxts':
            self.reference_files = self.oxts_files
            self.reference_timestamps = self.oxts_timestamps

        elif self.reference_sensor.startswith('cam'):
            cam_idx = int(self.reference_sensor.replace('cam', ''))
            img_key = f'image_{cam_idx:02d}'
            self.reference_files = [None] * len(self.image_timestamps[img_key])
            self.reference_timestamps = self.image_timestamps[img_key]

        else:
            raise ValueError(f"Unsupported reference sensor: {self.reference_sensor}")

    # ----------------------------------------------------------------------

    @cached_property
    def PointCloudShape(self):
        return (-1, 6)

    @cached_property
    def PointCloudDownSampleShape(self):
        return (50000, 6)

    @cached_property
    def ImageShape(self):
        return (1080, 1920, 3)

    @cached_property
    def OXTSShape(self):
        return (-1, 30)

    # ----------------------------------------------------------------------

    def _load_timestamps(self, filepath: str):
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            return [float(x.strip()) for x in f if x.strip()]

    def _find_closest_index(self, target, timestamps):
        if len(timestamps) == 0:
            return 0
        return int(np.argmin(np.abs(np.array(timestamps) - target)))

    def _process_point_cloud(self, pc):
        N = self.PointCloudDownSampleShape[0]
        if pc.shape[0] > N:
            idx = np.random.choice(pc.shape[0], N, replace=False)
            pc = pc[idx]
        else:
            pad = np.zeros((N - pc.shape[0], pc.shape[1]))
            pc = np.vstack((pc, pad))
        return pc

    # ----------------------------------------------------------------------
    #                          Video Frame Reader
    # ----------------------------------------------------------------------

    def _read_video_frame(self, img_key: str, global_index: int):
        """
        global_index: index in concatenated timestamp list → need to map to which sequence sub-video.
        """
        # Find which sub-video the frame belongs to
        total_frames_per_video = self.video_frame_counts[img_key]
        caps = self.video_caps[img_key]

        running = 0
        for vid_idx, frame_count in enumerate(total_frames_per_video):
            if global_index < running + frame_count:
                cap = caps[vid_idx]
                frame_in_video = global_index - running

                if cap is None:
                    return np.zeros(self.ImageShape, dtype=np.uint8)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
                ret, frame = cap.read()
                if not ret:
                    return np.zeros(self.ImageShape, dtype=np.uint8)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.ImageShape[1], self.ImageShape[0]))
                return frame

            running += frame_count

        # fallback
        return np.zeros(self.ImageShape, dtype=np.uint8)

    # ----------------------------------------------------------------------

    def __len__(self):
        return len(self.reference_timestamps)

    def __getitem__(self, idx):
        sample = {}
        ref_ts = self.reference_timestamps[idx]

        # ---------------- Velodyne ----------------
        if 'velodyne' in self.sensors and len(self.velodyne_files) > 0:
            li_idx = self._find_closest_index(ref_ts, self.velodyne_timestamps)
            pc = np.fromfile(self.velodyne_files[li_idx], dtype=np.float32).reshape(self.PointCloudShape)
            sample['velodyne'] = self._process_point_cloud(pc)

        # ---------------- Video Cameras ----------------
        for i in range(7):
            cam_key = f'cam{i}'
            img_key = f'image_{i:02d}'

            if cam_key in self.sensors:
                ts_list = self.image_timestamps[img_key]
                if len(ts_list) > 0:
                    closest = self._find_closest_index(ref_ts, ts_list)
                    img = self._read_video_frame(img_key, closest)
                    sample[img_key] = img

        # ---------------- OXTS ----------------
        if 'oxts' in self.sensors and len(self.oxts_files) > 0:
            oidx = self._find_closest_index(ref_ts, self.oxts_timestamps)
            with open(self.oxts_files[oidx], 'r') as f:
                arr = np.array([float(x) for x in f.readline().split()]).reshape(self.OXTSShape)
            sample['oxts'] = arr

        return sample
   

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    Only keep dataset_dir and seq_list.
    """
    parser = argparse.ArgumentParser(description="Hella Dataset Test")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default="/p/data1/nxtaim/proprietary/hella/HellaDataset/",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        '--seq_list',
        type=str,
        nargs='+',
        default=['seq1', 'seq2'],
        help="List of sequences to load"
    )
    parser.add_argument(
        '--sensors',
        type=str,
        nargs='+',
        default=['velodyne', 'oxts', 'cam0', 'cam6'],
        help="Sensors to include"
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to test HellaDataset WITHOUT DataLoader.
    """
    args = parse_args()

    # Initialize dataset
    dataset = HellaDataset(
        dataset_dir=args.dataset_dir,
        seq_list=args.seq_list,
        sensors=args.sensors
    )

    print(f"Dataset length = {len(dataset)}\n")

    # Directly read a few items
    for idx in range(min(5, len(dataset))):
        sample = dataset[idx]
        print(f"----- Sample {idx} -----")
        for key, val in sample.items():
            if isinstance(val, dict):
                print(f"{key}: dict-like")
            elif hasattr(val, 'shape'):
                print(f"{key}: shape={val.shape}")
            else:
                print(f"{key}: type={type(val)}")
        print("\n")


if __name__ == "__main__":
    main()