# In /p/project1/nxtaim-1/wunderlich3/valeo_dataloader/Faster_dataloader/preprocess_valeo_dataset.py

import os
import sys

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
# --- START: NEW UTILITY FUNCTION ---
@contextmanager
def suppress_stderr():
    """
    A context manager to temporarily redirect stderr to dev/null.
    This is useful for silencing C-level library warnings.
    """
    # On Windows, os.devnull is 'nul'; on Unix-like systems, it's '/dev/null'
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(sys.stderr.fileno())
    
    try:
        # Redirect stderr to the null device
        os.dup2(devnull_fd, sys.stderr.fileno())
        yield
    finally:
        # Restore the original stderr
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        os.close(devnull_fd)
        os.close(old_stderr_fd)
# --- END: NEW UTILITY FUNCTION ---


import argparse
from pathlib import Path
import cv2
cv2.setNumThreads(1)
import numpy as np
import polars as pl
from tqdm import tqdm
import h5py
import shutil

# --- Add this block to find the LiDM project ---
# This allows us to import the pcd2range utility function.
LIDM_PROJECT_PATH = "/p/project1/nxtaim-1/alagar1/LiDM-valeo/"
if LIDM_PROJECT_PATH not in sys.path:
    sys.path.append(LIDM_PROJECT_PATH)
    print(f"Added {LIDM_PROJECT_PATH} to Python path to import LiDM utilities.")

# --- Add this block to find the valeo_dataset project ---
# This allows us to import the ValeoDataset and its utilities.
# Assuming the valeo_dataset.py file is in the parent directory of this script.
VALEO_DATALOADER_PATH = str(Path(__file__).parent.parent)
if VALEO_DATALOADER_PATH not in sys.path:
    sys.path.append(VALEO_DATALOADER_PATH)
    print(f"Added {VALEO_DATALOADER_PATH} to Python path to import Valeo Dataloader.")

from lidm.utils.lidar_utils import pcd2range
from valeo_dataset import ValeoDataset, DatasetConfig, extract_frame_parquet, extract_frame_via_cv2

# --- START: CODE ADDED FROM OLD SCRIPT ---
# This helper function is taken directly from your old script to tile camera frames.
def create_video_grid(frames: list[np.ndarray], grid_shape: tuple[int, int]) -> np.ndarray:
    """Tiles a list of frames into a single grid image."""
    rows, cols = grid_shape
    if len(frames) != rows * cols:
        # Pad with black frames if necessary
        h, w, c = frames[0].shape
        black_frame = np.zeros((h, w, c), dtype=np.uint8)
        frames.extend([black_frame] * (rows * cols - len(frames)))

    h, w, c = frames[0].shape
    grid = np.zeros((h * rows, w * cols, c), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx * h : (row_idx + 1) * h, col_idx * w : (col_idx + 1) * w, :] = frame
        
    return grid
# --- END: CODE ADDED FROM OLD SCRIPT ---


def process_single_recording(recording_path: Path):
    """
    Runs the full in-place preprocessing pipeline on the recording directory.
    This includes metadata creation, video merging, and range image generation.
    """
    print("-" * 80)
    print(f"Processing recording: {recording_path}")

    # 1. Compute frame matches using the original ValeoDataset logic.
    #    This step reads the raw files and must be done first.
    config = DatasetConfig(
        base_path=str(recording_path),
        reference_sensor_id="HDL64E_ID_4",
        legacy_camera_filename=True,
        max_timestamp_diff_seconds=0.03,
        save_gnss_parquet=False,
    )
    # Use the base ValeoDataset, which reads raw files, not ValeoDatasetPreprocessed.
    original_dataset = ValeoDataset(config)
    
    # 2. Save metadata to metadata.parquet
    matches_df = pl.concat(original_dataset.matches)

    # Create a mapping of ref_frame_id to 0, 1, 2, 3...
    unique_ref_ids = matches_df["ref_frame_id"].unique(maintain_order=True)
    mapping_df = pl.DataFrame({
        "ref_frame_id": unique_ref_ids,
        "video_frame_idx": list(range(len(unique_ref_ids)))
    }).with_columns(pl.col("video_frame_idx").cast(pl.Int64))
    
    # Join this mapping back to the main metadata
    matches_df = matches_df.join(mapping_df, on="ref_frame_id", how="left")

    sensor_info = original_dataset.timestamps_df.select("sensor_id", "sensor_type").unique()
    matches_df_with_type = matches_df.join(sensor_info, on="sensor_id", how="left")
    metadata_path = recording_path / "metadata.parquet"
    matches_df_with_type.write_parquet(metadata_path)
    print(f"  -> Saved metadata for {len(original_dataset)} frames to {metadata_path.name}")

    # 3. Merge camera videos into a single tiled video
    print("  -> Merging camera videos...")
    cam_ids = sorted([sid for sid in original_dataset.data_paths_dict if sid.startswith("CAM_ID_")])
    grid_shape = (2, 3) # Standard 2x3 grid for 6 cameras

    if not cam_ids:
        print("  -> Warning: No cameras found to merge.")
    else:
        # Get one frame to determine video properties
        first_frame_tensor = extract_frame_via_cv2(
            original_dataset.data_paths_dict[cam_ids[0]]["data"], 0, order="HWC"
        )
        first_frame_np = ((first_frame_tensor.numpy() + 1.0) * 127.5).astype(np.uint8)
        h, w, _ = first_frame_np.shape
        
        # The merged video will be saved inside the 'camera' subdirectory.
        merged_video_path = recording_path / config.camera_dirname / "merged_cameras.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(merged_video_path), fourcc, 20.0, (w * grid_shape[1], h * grid_shape[0]))

        for i in tqdm(range(len(original_dataset)), desc="  Merging Videos", leave=False):
            match_group = original_dataset.matches[i]
            
            frames_to_tile = []
            for cam_id in cam_ids:
                cam_match = match_group.filter(pl.col("sensor_id") == cam_id)
                if cam_match.is_empty(): continue

                frame_idx = cam_match["frame_id"].item()
                video_path = original_dataset.data_paths_dict[cam_id]["data"]
                
                # Use the context manager to suppress low-level C library warnings
                with suppress_stderr():
                    # Extract as a normalized tensor [-1, 1]
                    frame_tensor = extract_frame_via_cv2(video_path, frame_idx, order="HWC", colorspace="BGR")
                    # Convert back to standard image range [0, 255]
                    frame = ((frame_tensor.numpy() + 1.0) * 127.5).astype(np.uint8)
                
                frames_to_tile.append(frame)
            
            if frames_to_tile:
                grid_frame = create_video_grid(frames_to_tile, grid_shape)
                video_writer.write(grid_frame)
            
        video_writer.release()
        print(f"  -> Saved merged video to {merged_video_path.relative_to(recording_path)}")

    # 4. Pre-calculate and save Velodyne range images to an HDF5 file
    print("  -> Generating Velodyne range images...")
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
        for i in tqdm(range(len(original_dataset)), desc="  Generating Range Images", leave=False):
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
    root_path = Path(args.dataset_root_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"The specified root path does not exist: {root_path}")

    print("Searching for recording directories...")
    all_paths = []
    # 1. FIND AND FILTER PATHS FIRST
    for bin_path in root_path.rglob("bin"):
        if not bin_path.is_dir(): continue
        for recording_path in bin_path.iterdir():
            if recording_path.is_dir():
                # Check for required subfolders
                if not (recording_path / "camera").exists() or not (recording_path / "velodyne").exists():
                    continue
                # Check for LiDAR files
                if not list((recording_path / "velodyne").glob("*HDL64E_ID_4.parquet")):
                    continue
                all_paths.append(recording_path)
    
    all_paths = sorted(list(set(all_paths)))
    
    if not all_paths:
        print(f"Error: No valid recording folders found in '{root_path}'.")
        return

    print(f"Found {len(all_paths)} valid recording directories to process.")

    # 2. PARALLEL EXECUTION
    # Adjusted to 12. If it crashes with "Out of Memory", lower this to 4 or 8.
    MAX_CONCURRENT_RECORDINGS = 12 
    print(f"Starting parallel processing with {MAX_CONCURRENT_RECORDINGS} workers...")


    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_RECORDINGS) as executor:
        # Submit all tasks to the pool
        future_to_path = {executor.submit(process_single_recording, path): path for path in all_paths}
        
        # Use tqdm to track the progress of the futures as they complete
        for future in tqdm(as_completed(future_to_path), total=len(all_paths), desc="Overall Progress"):
            path = future_to_path[future]
            try:
                # This will re-raise any exception caught inside the worker process
                future.result() 
            except Exception as e:
                print(f"\n!! FAILED to process directory: {path} !!")
                print(f"  -> Error: {e}")
                # Optional: print traceback for deep debugging
                # import traceback; traceback.print_exc()

    print("\nAll recordings have been processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valeo Dataset Batch Preprocessing Script")
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        required=True,
        help="Path to the top-level dataset directory containing the 'bin' subdirectories (e.g., 'test_space').",
    )
    args = parser.parse_args()
    main(args)
