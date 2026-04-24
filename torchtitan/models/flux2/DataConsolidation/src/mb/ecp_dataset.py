import os
import json
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)


def group_paths_by_directory(paths):
    """
    Groups a list of file paths into a list of lists, where each inner list
    contains paths that share the same parent directory.

    Args:
        paths (list): A list of file paths (strings).

    Returns:
        list: A list of lists, with each inner list containing paths from
              the same directory.
    """
    grouped_by_dir = defaultdict(list)

    for path in paths:
        directory = os.path.dirname(path)
        grouped_by_dir[directory].append(path)

    return list(map(sorted, grouped_by_dir.values()))


def extract_frames_via_cv2(
    video_path, start_frame_idx, frame_interval, frame_step, blur=False
):
    """
    Extracts frames from a video file using OpenCV.

    Parameters:
        video_path (str): Path to the video file.
        start_frame_idx (int): Index of the first frame to start extraction.
        frame_interval (int): Number of frames to extract.
        frame_step (int): Step size between frames to be extracted.

    Returns:
        np.ndarray: Array of extracted frames.

    Raises:
        ValueError: If a frame cannot be read from the video file.

    Example:
        >>> frames = extract_frames_via_cv2('video.mp4', 0, 10, 2)
        >>> print(frames.shape)
        (10, height, width, channels)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)  # Set the starting frame position
    frames = []
    idx = -1

    while len(frames) < frame_interval:
        ret, frame = cap.read()
        idx += 1

        if idx % frame_step:  # Skip frames based on the frame_step
            continue

        if not ret:
            raise ValueError(
                f"Could not read frame {start_frame_idx + idx} from {video_path}"
            )

        if blur:
            frame = cv2.GaussianBlur(frame, (41, 41), 0)

        frames.append(frame)

    cap.release()
    return np.array(frames)


class ECPDataset(Dataset):
    """
    A dataset class for loading video frames and associated metadata using OpenCV.

    Parameters:
        video_dir (str): Directory containing video files.
        frame_interval (int): Number of frames to extract per video. Default is 1.
        frame_step (int): Step size between frames to be extracted. Default is 1.
        size (tuple): Desired size (height, width) to resize frames. Default is None.
        transform (callable): Transformations to apply to the frames. Default is None.

    Attributes:
        video_paths (list): List of lists of video file paths, grouped by directory.
                            Each inner list represents a multi-camera setup for a single scene.
        fpss (np.ndarray): Array of frames per second for each video group (taken from the first camera in the group).
        frames (np.ndarray): Array of total frames for each video group (taken from the first camera in the group).
        widths (np.ndarray): Array of frame widths for each video group (taken from the first camera in the group).
        heights (np.ndarray): Array of frame heights for each video group (taken from the first camera in the group).
        metadata (dict): Dictionary containing *full* metadata lists for each video group,
                         keyed by the path of the first camera in that group. This is because
                         the metadata is a JSONL file that needs to be loaded entirely.

    Methods:
        __len__(): Returns the total number of (video_group, frame_interval) combinations in the dataset.
        __getitem__(idx): Returns a dictionary with multi-camera video frames and metadata for the given index.
        save_meta_to_parquet(filepath): Saves the video metadata to a CSV or Parquet file.
    """

    def __init__(
        self,
        video_dir,
        frame_interval=1,
        frame_step=1,
        size=None,
        transform=None,
    ):
        # Collect all relevant video paths first
        all_raw_video_paths = [str(path) for path in Path(video_dir).rglob("*.mp4")]

        # Group them by directory. self.video_paths is now a list of lists.
        self.video_paths = group_paths_by_directory(all_raw_video_paths)

        self.fpss = []
        self.frames = []
        self.widths = []
        self.heights = []

        # Store metadata keyed by the first video path in each group
        # This will store the *full* list of dictionaries from the JSONL file
        self.metadata = {}

        print("Gathering video data and metadata...")
        # Iterate through each group of videos
        for group_idx, video_group in enumerate(tqdm(self.video_paths)):
            if not video_group:  # Skip empty groups if any
                rank_zero_print(
                    f"Warning: Empty video group found at index {group_idx}. Skipping."
                )
                continue

            # For properties like fps, frames, width, height, we assume they are consistent
            # across cameras within the same group and take them from the first video.
            first_video_path_in_group = video_group[0]

            try:
                cap = cv2.VideoCapture(first_video_path_in_group)
                if not cap.isOpened():
                    raise IOError(
                        f"Could not open video file: {first_video_path_in_group}"
                    )

                self.fpss.append(cap.get(cv2.CAP_PROP_FPS))
                self.frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.widths.append(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.heights.append(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception as e:
                rank_zero_print(
                    f"Got an exception when processing '{first_video_path_in_group}' for group properties.\nError:{e}"
                )
                # Handle error: remove this group from consideration or mark it invalid
                self.fpss.append(np.nan)
                self.frames.append(0)  # Mark as 0 frames to prevent further processing
                self.widths.append(0)
                self.heights.append(0)
                continue  # Skip metadata loading for this problematic group

            # Load metadata once per group, using the first video's path to locate metadata_flexray_synced.json
            metadata_path = os.path.join(
                os.path.dirname(first_video_path_in_group),
                "metadata_flexray_synced.json",
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    # Load all lines from the JSONL file
                    full_metadata_list = [json.loads(line) for line in f]
                self.metadata[first_video_path_in_group] = full_metadata_list
            else:
                self.metadata[first_video_path_in_group] = (
                    []
                )  # Store an empty list if no metadata
                rank_zero_print(
                    f"No metadata_flexray_synced.json found for group starting with '{first_video_path_in_group}'."
                )

        self.fpss = np.array(self.fpss)
        self.frames = np.array(self.frames, dtype=int)
        self.widths = np.array(self.widths, dtype=int)
        self.heights = np.array(self.heights, dtype=int)

        self.frame_interval = frame_interval
        self.frame_step = frame_step

        self.transform = transform
        if size is not None:
            assert (
                hasattr(size, "__len__") and len(size) == 2
            ), f"size needs to be a tuple, instead got: {size}"
            resize = v2.Resize((size[0], size[1]), antialias=True)
            if self.transform is None:
                self.transform = resize
            else:
                self.transform = v2.Compose([resize, self.transform])

        # Now, idx_map stores (group_idx, start_frame_idx)
        # self.idx_map = []
        # for group_idx, num_frame in enumerate(self.frames):
        #     if num_frame > 0:  # Only process groups with valid frame counts
        #         for x in range(0, num_frame // ((frame_interval - 1) * frame_step + 1)):
        #             self.idx_map.append(
        #                 (group_idx, x * ((frame_interval - 1) * frame_step + 1))
        #             )
        frames_needed_per_sequence = (frame_interval - 1) * frame_step + 1
        zero = np.array([0], dtype=int)
        self.cum_avail_startframes = np.cumsum(np.maximum(self.frames-frames_needed_per_sequence+1, zero))
        self.cum_startframes_before = np.concatenate([zero, self.cum_avail_startframes[:-1]])


    def __len__(self):
        return self.cum_avail_startframes[-1]

    def _get_metadata(self, group_idx, start_frame_idx, frame_interval, frame_step):
        """
        Retrieves metadata aligned with the video frames for a specific video group.
        This function now slices from the pre-loaded full metadata list.

        Parameters:
            group_idx (int): Index of the video group.
            start_frame_idx (int): Starting frame index for extraction.
            frame_interval (int): Number of frames to extract.
            frame_step (int): Step size between frames to be extracted.

        Returns:
            dict: Dictionary containing aligned metadata for the video frames.
        """
        # Use the first video path in the group as the key for metadata
        first_video_path_in_group = self.video_paths[group_idx][0]

        if (
            first_video_path_in_group in self.metadata
            and self.metadata[first_video_path_in_group]
        ):
            full_metadata_list = self.metadata[first_video_path_in_group]

            # Apply slicing to the pre-loaded full metadata list
            end_idx = start_frame_idx + frame_interval * frame_step + 1

            # Handle potential out-of-bounds slicing for metadata
            # if end_idx > len(full_metadata_list):
            #     rank_zero_print(f"Warning: Metadata list for group {group_idx} is too short. "
            #                     f"Requested up to index {end_idx-1}, but list has length {len(full_metadata_list)}. "
            #                     f"Slicing up to available length.")
            #     sliced_metadata = full_metadata_list[start_frame_idx : len(full_metadata_list) : frame_step]
            # else:
            sliced_metadata = full_metadata_list[start_frame_idx:end_idx:frame_step]

            if not sliced_metadata:  # If slicing resulted in an empty list
                return {}

            # Restructure the sliced metadata from a list of dicts to a dict of lists
            # Example: [{'a':1, 'b':5}, {'a':2, 'b':6}] -> {'a':[1,2], 'b':[5,6]}
            return {
                key: [m[key] for m in sliced_metadata] for key in sliced_metadata[0]
            }
        else:
            return {}

    def __getitem__(self, idx):
        """
        Retrieves multi-camera video frames and associated metadata for the given index.

        Parameters:
            idx (int): Index of the (video_group, start_frame_idx) combination in the dataset.

        Returns:
            dict: Dictionary containing video frames (stacked along a new camera axis), and metadata.

        Raises:
            Exception: If an error occurs while processing the video group.
        """
        try:
            # group_idx, start_frame_idx = self.idx_map[idx]
            group_idx = int(np.searchsorted(self.cum_avail_startframes, idx, side="right"))
            start_frame_idx = idx - self.cum_startframes_before[group_idx]
            video_group_paths = self.video_paths[group_idx]

            video_group_paths = self.video_paths[group_idx]

            all_camera_frames = []
            for camera_path in video_group_paths:
                frames = extract_frames_via_cv2(
                    camera_path,
                    start_frame_idx,
                    self.frame_interval,
                    self.frame_step,
                )
                # Convert to torch tensor, change color channel order (BGR to RGB if cv2), and permute dimensions (T, H, W, C) to (T, C, H, W)
                frames = torch.from_numpy(
                    np.ascontiguousarray(frames[:, :, :, ::-1].transpose(0, 3, 1, 2))
                )

                if self.transform is not None:
                    # Apply transform to each camera's frames
                    frames = self.transform(frames)

                all_camera_frames.append(frames)

            # Stack frames from all cameras along a new dimension (camera axis)
            # Resulting shape: (num_cameras, num_frames_per_interval, C, H, W)
            final_frames = torch.stack(all_camera_frames, dim=0)

            # Retrieve metadata for the group
            metadata = self._get_metadata(
                group_idx, start_frame_idx, self.frame_interval, self.frame_step
            )

            # Add additional metadata fields
            metadata["video_group_paths"] = (
                video_group_paths  # Store all paths for reference
            )
            metadata["start_frame_idx"] = start_frame_idx
            metadata["frame_interval"] = self.frame_interval
            metadata["frame_step"] = self.frame_step

            # Use the dimensions of the processed frames for aspect ratio, width, height
            # Assuming final_frames shape is (num_cameras, num_frames_per_interval, C, H, W)
            if final_frames.shape[3] > 0:  # Check if height is valid
                metadata["aspect_ratio"] = (
                    final_frames.shape[4] / final_frames.shape[3]
                )  # W / H
            else:
                metadata["aspect_ratio"] = 0.0  # Or handle as error
            metadata["orig_width"] = final_frames.shape[4]  # W
            metadata["orig_height"] = final_frames.shape[3]  # H

            return {"images": final_frames, "metadata": metadata}
        except Exception as e:
            group_idx, start_frame_idx = self.idx_map[idx]  # Re-get for error message
            rank_zero_print(
                f"Got error at getting video group {group_idx} (idx_map[{idx}]), "
                f"start_frame_idx {start_frame_idx}",
                "\nError:",
                e,
            )
            raise e


def custom_collate_fn(batch):
    """
    Custom collate function for the TrafficLightDataset.
    It correctly batches images and collects metadata lists as lists of lists.

    Args:
        batch (list): A list of samples, where each sample is a dictionary
                      returned by TrafficLightDataset.__getitem__.

    Returns:
        dict: A dictionary containing batched images and correctly structured metadata.
    """
    batched_images = []
    batched_metadata = defaultdict(list)

    for item in batch:
        # Collect images (assuming they are already torch.Tensors)
        batched_images.append(item["images"])

        # Collect metadata. For each key, append the entire list/value
        # from the current sample's metadata.
        for key, value in item["metadata"].items():
            batched_metadata[key].append(value)

    # Stack the images along a new batch dimension
    # Assuming images are (num_cameras, num_frames_per_interval, C, H, W)
    final_images = torch.stack(batched_images, dim=0)

    # Convert defaultdict to a regular dict
    final_metadata = dict(batched_metadata)

    return {"images": final_images, "metadata": final_metadata}


def save_dataloader_iteration_as_video(
    data_item: dict,
    output_video_path: str,
    fps: int = 10,
    codec: str = "mp4v",  # Default for .mp4. Use "XVID" for .avi, "MJPG" for .avi (larger files)
    is_color: bool = True,
    label_font_scale: float = 0.7,
    label_thickness: int = 2,
    label_color: tuple = (0, 255, 0),  # Green in BGR format
):
    """
    Saves a single iteration of the dataloader (multi-camera frames) as a video file.
    The video displays frames from multiple cameras side-by-side or in quadrants.

    Args:
        data_item (Dict[str, Any]): The output dictionary from one dataloader iteration,
                                     containing 'images' and 'metadata'.
                                     'images' should be a tensor of shape
                                     (num_cameras, num_frames_per_interval, C, H, W).
        output_video_path (str): The path to save the output video file (e.g., "output.mp4").
        fps (int): Frames per second for the output video.
        codec (str): FourCC code for the video codec (e.g., "mp4v" for .mp4, "XVID" for .avi).
        is_color (bool): True if the video frames are color, False for grayscale.
        label_font_scale (float): Font scale for camera labels.
        label_thickness (int): Thickness for camera labels.
        label_color (tuple): BGR color for camera labels.
    """
    if "images" not in data_item or "metadata" not in data_item:
        raise ValueError("data_item must contain 'images' and 'metadata' keys.")

    final_frames_tensor = data_item[
        "images"
    ]  # Shape: (num_cameras, num_frames_per_interval, C, H, W)
    metadata = data_item["metadata"]

    num_cameras = final_frames_tensor.shape[0]
    num_frames_per_interval = final_frames_tensor.shape[1]
    C, H, W = final_frames_tensor.shape[2:]

    # Get camera names for labels.
    # We try to extract a more descriptive name from the path if available.
    camera_paths = metadata.get(
        "video_group_paths", [f"Camera {i}" for i in range(num_cameras)]
    )
    camera_names = []
    for p in camera_paths:
        # Example: "path/to/ZyncBoxCamera_10200.mp4" -> "ZyncBoxCamera_10200"
        # Or "path/to/ampel_video.mp4" -> "ampel_video"
        camera_names.append(os.path.splitext(os.path.basename(p))[0])

    # Fallback if camera_names are not descriptive or missing
    if not camera_names or len(camera_names) != num_cameras:
        camera_names = [f"Camera {i}" for i in range(num_cameras)]

    # Determine output video dimensions and layout based on number of cameras
    output_width, output_height = W, H
    layout_grid = (1, 1)  # Default for 1 camera

    if num_cameras == 2:
        layout_grid = (1, 2)  # Side-by-side
        output_width = 2 * W
    elif num_cameras == 3 or num_cameras == 4:
        layout_grid = (2, 2)  # 2x2 grid
        output_width = 2 * W
        output_height = 2 * H
    elif num_cameras > 4:
        # For more than 4 cameras, stack them vertically for simplicity
        rank_zero_print(
            f"Warning: More than 4 cameras ({num_cameras}). Stacking vertically."
        )
        layout_grid = (num_cameras, 1)
        output_width = W
        output_height = num_cameras * H

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (output_width, output_height), is_color
    )

    if not out.isOpened():
        raise IOError(
            f"Could not open video writer for path: {output_video_path}. "
            f"Check codec '{codec}' and file extension."
        )

    for frame_idx in range(num_frames_per_interval):
        processed_frames_for_this_interval = []
        for cam_idx in range(num_cameras):
            # Extract frame, convert to numpy, permute (C,H,W) to (H,W,C)
            # .cpu().numpy() moves tensor to CPU and converts to NumPy array
            frame_np = np.ascontiguousarray(
                final_frames_tensor[cam_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
            )

            # Assuming frames are normalized [0,1] and RGB from torchvision transforms
            # Convert to BGR and scale to [0, 255] for OpenCV
            # *Add other transforms code if applied in dataloader
            if frame_np.max() <= 1:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Add camera label
            label_text = camera_names[cam_idx]
            cv2.putText(
                frame_np,
                label_text,
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                label_color,
                label_thickness,
                cv2.LINE_AA,
            )
            processed_frames_for_this_interval.append(frame_np)

        # Arrange frames based on layout
        final_display_frame = None
        if num_cameras == 1:
            final_display_frame = processed_frames_for_this_interval[0]
        elif num_cameras == 2:
            final_display_frame = cv2.hconcat(processed_frames_for_this_interval)
        elif num_cameras == 3:
            # Pad the 4th quadrant with a black frame
            blank_frame = np.zeros((H, W, C), dtype=np.uint8)
            # Ensure blank frame is BGR if is_color is True
            if is_color and C == 3:
                blank_frame = cv2.cvtColor(
                    blank_frame, cv2.COLOR_RGB2BGR
                )  # It's already BGR if initialized with 3 channels

            top_row = cv2.hconcat(
                [
                    processed_frames_for_this_interval[0],
                    processed_frames_for_this_interval[1],
                ]
            )
            bottom_row = cv2.hconcat(
                [processed_frames_for_this_interval[2], blank_frame]
            )
            final_display_frame = cv2.vconcat([top_row, bottom_row])
        elif num_cameras == 4:
            top_row = cv2.hconcat(
                [
                    processed_frames_for_this_interval[0],
                    processed_frames_for_this_interval[1],
                ]
            )
            bottom_row = cv2.hconcat(
                [
                    processed_frames_for_this_interval[2],
                    processed_frames_for_this_interval[3],
                ]
            )
            final_display_frame = cv2.vconcat([top_row, bottom_row])
        elif num_cameras > 4:
            # Stack vertically
            final_display_frame = cv2.vconcat(processed_frames_for_this_interval)

        if final_display_frame is not None:
            out.write(final_display_frame)
        else:
            rank_zero_print(
                f"Warning: No frame generated for frame_idx {frame_idx} with {num_cameras} cameras."
            )

    out.release()
    rank_zero_print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    dataset_root = "Euro-cityAutobahn-scenarioPolice_2017-04-30-12-44-31"
    dataset = ECPDataset(dataset_root, frame_interval=50, frame_step=2)
    loader = DataLoader(dataset, 2, False, collate_fn=custom_collate_fn)
    batched_data = next(iter(loader))
    # data = dataset[0]
    # save_dataloader_iteration_as_video(data, "ecp_sample.mp4", fps=7)
