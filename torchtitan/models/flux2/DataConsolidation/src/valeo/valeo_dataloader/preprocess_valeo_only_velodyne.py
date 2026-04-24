# In /p/project1/nxtaim-1/wunderlich3/valeo_dataloader/Faster_dataloader/preprocess_valeo_dataset.py

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
import h5py

# --- Add this block to find the LiDM project ---
LIDM_PROJECT_PATH = "/p/project1/nxtaim-1/wunderlich3/LiDM-valeo/"
if LIDM_PROJECT_PATH not in sys.path:
    sys.path.append(LIDM_PROJECT_PATH)
    print(f"Added {LIDM_PROJECT_PATH} to Python path to import LiDM utilities.")

from lidm.utils.lidar_utils import pcd2range
from valeo_dataset import ValeoDatasetPreprocessed, DatasetConfig, extract_frame_parquet

def process_single_recording(recording_path: Path):
    """
    Runs the full in-place preprocessing pipeline on a single recording directory.
    """
    print("-" * 80)
    print(f"Processing recording: {recording_path}")

    # 1. Compute frame matches using the original ValeoDatasetPreprocessed logic
    config = DatasetConfig(
        base_path=str(recording_path),
        reference_sensor_id="HDL64E_ID_4",
        legacy_camera_filename=True,
        max_timestamp_diff_seconds=0.03,
    )
    original_dataset = ValeoDatasetPreprocessed(config)
    
    # Save metadata
    matches_df = pl.concat(original_dataset.matches)
    sensor_info = original_dataset.timestamps_df.select("sensor_id", "sensor_type").unique()
    matches_df_with_type = matches_df.join(sensor_info, on="sensor_id", how="left")
    metadata_path = recording_path / "metadata.parquet"
    matches_df_with_type.write_parquet(metadata_path)
    print(f"  -> Saved metadata for {len(original_dataset)} frames.")

    # 2. Pre-calculate and save range images to an HDF5 file
    img_size = (64, 1024)
    fov = (3, -25)
    depth_range = (1.0, 56.0)
    h5_path = recording_path / "velodyne_range_images.h5"
    
    with h5py.File(h5_path, 'w') as hf:
        dset = hf.create_dataset(
            "range_images",
            shape=(len(original_dataset), img_size[0], img_size[1]),
            dtype='f4',
            compression="gzip"
        )
        for i in range(len(original_dataset)):
            match_group = original_dataset.matches[i]
            velodyne_match = match_group.filter(pl.col("sensor_id") == "HDL64E_ID_4")
            if velodyne_match.is_empty(): continue
            
            frame_id = velodyne_match["frame_id"].item()
            pcd_path = original_dataset.data_paths_dict["HDL64E_ID_4"]["data"]
            
            pcd_tensor = extract_frame_parquet(pcd_path, frame_id)
            xyz_points = pcd_tensor[:, :3].numpy()
            
            proj_range, _ = pcd2range(xyz_points, img_size, fov, depth_range)
            dset[i, :, :] = proj_range
            
    print(f"  -> Saved range images to {h5_path.name}")
    print(f"Finished processing: {recording_path}")

def main(args):
    """
    Finds all recording subdirectories based on the expected structures and processes them.
    """
    root_path = Path(args.dataset_root_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"The specified root path does not exist: {root_path}")

    # --- THIS IS THE MODIFIED SECTION ---
    # Recursively search for all directories named 'bin' at any level,
    # then collect their immediate subdirectories.
    print("Searching for recording directories...")
    all_paths = []
    # Use rglob('**/bin') to find all 'bin' directories recursively
    for bin_path in root_path.rglob("bin"):
        if not bin_path.is_dir():
            continue
        # A recording path is a subdirectory inside a 'bin' folder
        for recording_path in bin_path.iterdir():
            if recording_path.is_dir():
                all_paths.append(recording_path)
    
    # Combine the lists and remove duplicates by converting to a set, then back to a sorted list.
    all_paths = sorted(list(set(all_paths)))
    # --- END OF MODIFIED SECTION ---
    
    if not all_paths:
        print(f"Error: No recording folders found in '{root_path}' matching the patterns '*/bin/*' or '*/*/bin/*'.")
        print("Please check your directory structure.")
        return

    print(f"Found {len(all_paths)} recording directories to process.")
    
    for path in tqdm(all_paths, desc="Overall Progress"):
        if not (path / "camera").exists() or not (path / "velodyne").exists():
            print(f"\nSkipping directory {path} as it's missing 'camera' or 'velodyne' subfolders.")
            continue
        
        try:
            process_single_recording(path)
        except Exception as e:
            print(f"\n!! FAILED to process directory: {path} !!")
            print(f"  -> Error: {e}")
            print("  -> Skipping to the next directory.")
            import traceback
            traceback.print_exc()
            continue

    print("\nAll recordings have been processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valeo Dataset Batch Preprocessing Script")
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        required=True,
        help="Path to the top-level dataset directory (e.g., 'test_space').",
    )
    args = parser.parse_args()
    main(args)
