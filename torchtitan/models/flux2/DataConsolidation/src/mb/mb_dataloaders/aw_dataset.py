from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import cv2

def read_split_file(split):
    idx_map = []
    with open(split) as fr:
        for line in fr:
            idx_map.append((line.split(',')[0],line.split(',')[1].replace('\n','')))



    return idx_map







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
        # >>> frames = extract_frames_via_cv2('video.mp4', 0, 10, 2)
        # >>> print(frames.shape)
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



class AWDataset(Dataset):
    """
    PyTorch Dataset for loading video frames from a directory of videos.

    All videos are expected to be in a single directory. The dataset extracts
    the same number of frames from each video and returns them as torch tensors.
    """

    def __init__(
            self,
            video_dir,
            frame_interval=10,
            frame_step=1,
            start_map_idx=0,
            blur=False,
            transform=None,
            metadata_func=None,
    ):
        """
        Args:
            idx_map (list of tuples): contains video paths and start frame indexes.
            frame_interval (int): Number of frames to extract from each video.
            frame_step (int): Step size between frames to be extracted.
            blur (bool): Whether to apply Gaussian blur to frames.
            transform (callable, optional): Optional transform to be applied to frames.
            metadata_func (callable, optional): Function to extract metadata for frames.
                Should have signature: metadata_func(video_path, frame_indices) and return a dict.
        """
        self.video_dir = video_dir
        self.frame_interval = frame_interval
        self.frame_step = frame_step
        self.start_map_idx = start_map_idx
        self.blur = blur
        self.transform = transform
        self.metadata_func = metadata_func
        self.idx_map = self.create_idx_map()

        if len(self.idx_map) == 0:
            raise ValueError(f"No .mp4 files found in {video_dir}")

    def __len__(self):
        """Returns the number of videos in the dataset."""
        return len(self.idx_map)

    def __getitem__(self, idx):
        """
        Extracts frames from a video and returns them as a dictionary.

        Args:
            idx (int): Index of the video to load.

        Returns:
            dict: Dictionary with keys:
                - "images": Tensor of shape (1, frame_interval, 3, height, width) for single camera
                - "metadata": Dictionary containing aligned metadata for the frames
        """

        video_path = self.idx_map[idx][0]
        start_frame_idx = int(self.idx_map[idx][1])
        frames = self._extract_frames(video_path, start_frame_idx)

        # Get original dimensions
        orig_width = frames.shape[2]
        orig_height = frames.shape[1]

        # Convert to torch tensor, change color channel order (BGR to RGB), and permute dimensions
        # (T, H, W, C) to (T, C, H, W)
        frames_tensor = torch.from_numpy(
            np.ascontiguousarray(frames[:, :, :, ::-1].transpose(0, 3, 1, 2))
        ).float()

        # Add camera dimension: (1, frame_interval, 3, height, width)
        frames_tensor = frames_tensor.unsqueeze(0)

        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        # Get metadata for the video
        metadata = self._get_metadata(idx, orig_width, orig_height, start_frame_idx)

        return {
            "images": frames_tensor,
            "metadata": metadata
        }

    def create_idx_map(self):
        video_paths = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if
                       os.path.isfile(os.path.join(self.video_dir, f)) and f.endswith('.mp4')]
        video_counter = 0
        idx_map = []
        frames_needed = self.frame_step * self.frame_interval

        for video_path in tqdm(video_paths):
            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()

            if self.start_map_idx + frames_needed <= frame_count:
                video_counter += 1

            for start_frame in range(self.start_map_idx, frame_count, frames_needed):
                if start_frame + frames_needed <= frame_count:
                    idx_map.append((video_path, start_frame))

        print(
            f"Found {len([os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if os.path.isfile(os.path.join(self.video_dir, f)) and f.endswith('.mp4')])} over all. Usable videos found {video_counter}. "
            f"Assorted {len([os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if os.path.isfile(os.path.join(self.video_dir, f)) and f.endswith('.mp4')]) - video_counter}")

        return idx_map

    def _extract_frames(self, video_path, start_frame_idx):
        """
        Extracts frames from a video file using the provided extraction function.

        Args:
            video_path (Path or str): Path to the video file.

        Returns:
            np.ndarray: Array of extracted frames with shape (frame_interval, H, W, 3).
        """
        return extract_frames_via_cv2(
            video_path,
            start_frame_idx,
            self.frame_interval,
            self.frame_step,
            self.blur
        )

    def _get_metadata(self, idx, orig_width, orig_height, start_frame_idx):
        """
        Retrieves aligned metadata for the extracted video frames.

        Args:
            idx (int): Index of the video.
            orig_width (int): Original width of the video frames.
            orig_height (int): Original height of the video frames.

        Returns:
            dict: Dictionary containing metadata aligned with the extracted frames.
        """
        video_path = self.idx_map[idx][0]

        return {
            'video_path': str(video_path),
            'start_frame_idx': start_frame_idx,
            'frame_interval': self.frame_interval,
            'frame_step': self.frame_step,
            'aspect_ratio': orig_width / orig_height,
            'orig_width': orig_width,
            'orig_height': orig_height
        }



if __name__ == "__main__":
    start_map_idx = 0
    frame_step = 2
    frame_interval = 50
    dataset_root = "/home/alzuber/external/ext1/videos"

    dataset = AWDataset(
        video_dir=dataset_root,
        start_map_idx=start_map_idx,
        frame_interval=frame_interval,
        frame_step=frame_step,
        blur=True,
    )

    loader = DataLoader(dataset, 2, False)
    for x in tqdm(loader):
        pass
    print("Succes")
    # batched_data = next(iter(loader))
    # data = dataset[1]
    # save_dataloader_iteration_as_video(data, "test.mp4", fps=7)
