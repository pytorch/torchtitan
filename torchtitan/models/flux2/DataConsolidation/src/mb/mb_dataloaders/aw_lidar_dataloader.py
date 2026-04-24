from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os





def normalize_pointcloud_size(pointcloud, target_num_points):
    """
    Normalize pointcloud to have exactly target_num_points.

    Args:
        pointcloud: numpy array of shape (num_points, 3)
        target_num_points: desired number of points

    Returns:
        numpy array of shape (target_num_points, 3)
    """
    current_num_points = pointcloud.shape[0]

    if current_num_points > target_num_points:
        # Randomly sample points
        indices = np.random.choice(current_num_points, target_num_points, replace=False)
        return pointcloud[indices]
    elif current_num_points < target_num_points:
        # Pad with zeros
        padding = np.zeros((target_num_points - current_num_points, pointcloud.shape[1]))
        return np.vstack([pointcloud, padding])
    else:
        return pointcloud



class AWLidarDataset(Dataset):
    """
    PyTorch Dataset for loading video frames from a directory of videos.

    All videos are expected to be in a single directory. The dataset extracts
    the same number of frames from each video and returns them as torch tensors.
    """

    def __init__(
            self,
            lidar_dir,
            interval=10,
            step=1,
            start_idx=0,
            points=1000,
            transform=None,
            metadata_func=None,
    ):
        """
        Args:
            video_dir (str): Path to directory containing .mp4 video files.
            frame_interval (int): Number of frames to extract from each video.
            frame_step (int): Step size between frames to be extracted.
            start_frame_idx (int): Index of the first frame to start extraction.
            blur (bool): Whether to apply Gaussian blur to frames.
            transform (callable, optional): Optional transform to be applied to frames.
            metadata_func (callable, optional): Function to extract metadata for frames.
                Should have signature: metadata_func(video_path, frame_indices) and return a dict.
        """

        self.lidar_dir = Path(lidar_dir)
        self.interval = interval
        self.step = step
        self.start_idx = start_idx
        self.transform = transform
        self.metadata_func = metadata_func
        self.points = points

        # Collect all relevant pointcloud paths first
        self.lidar_paths = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if os.path.isdir(os.path.join(lidar_dir, f))]

        self.idx_map = []
        pcds_needed = interval * step

        for dir_idx, lidar_path in enumerate(self.lidar_paths):
            if len(os.listdir(lidar_path)) < pcds_needed:
                continue
            else:
                for start_idx in range(0, len(os.listdir(lidar_path)), pcds_needed):
                    if start_idx +  pcds_needed <= len(os.listdir(lidar_path)) - 1:
                        self.idx_map.append((dir_idx, start_idx))





        if len(self.lidar_paths) == 0:
            raise ValueError(f"No .mp4 files found in {lidar_dir}")

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
        dir_idx, start_idx = self.idx_map[idx]

        rec_path = self.lidar_paths[dir_idx]
        rec_pcds = [os.path.join(rec_path, f) for f in os.listdir(rec_path) if f.endswith(".npy")]

        pcds = []
        i = start_idx
        while len(pcds) < self.interval:

            # Skip pcds


            pcd = rec_pcds[i]
            loaded_pcd = np.load(pcd)
            if loaded_pcd.shape[0] < self.points:
                processsed_pcd = np.pad(loaded_pcd, ((0, self.points - loaded_pcd.shape[0]), (0,0)), mode='constant', constant_values=0)
            else:
                processsed_pcd = normalize_pointcloud_size(loaded_pcd, self.points)

            # print(f"Percentage of displayed points: {(self.points / loaded_pcd.shape[0]) * 100}%")

            # o3d_pcd = open3d.geometry.PointCloud()
            # o3d_pcd.points = open3d.utility.Vector3dVector(processsed_pcd)
            # open3d.visualization.draw_geometries([o3d_pcd])

            pcds.append(processsed_pcd)

            i += self.step

        tensor = torch.from_numpy(np.stack(pcds))

        metadata = self._get_metadata(dir_idx, start_idx)

        return {
            'pointclouds' : tensor,
            'metadata' : metadata
        }




    def _get_metadata(self, dir_idx, start_idx):
        """
        Retrieves aligned metadata for the extracted video frames.

        Args:
            idx (int): Index of the video.
            orig_width (int): Original width of the video frames.
            orig_height (int): Original height of the video frames.

        Returns:
            dict: Dictionary containing metadata aligned with the extracted frames.
        """

        pcd_path = self.lidar_paths[dir_idx]

        return {
            'lidar_path': str(pcd_path),
            'start_idx': start_idx,
            'interval': self.interval,
            'step': self.step,
            'number_of_points': self.points,
        }

    def get_video_path(self, idx):
        """Returns the file path of the video at the given index."""
        return self.video_paths[idx]


if __name__ == "__main__":
    dataset_root = "/home/alzuber/external/ext0/pointclouds"
    dataset = AWLidarDataset(
        lidar_dir=dataset_root,
        interval=50,
        step=5,
        points=55000,
    )

    loader = DataLoader(dataset, 2, False)
    for x in tqdm(loader):
        pass
    print("Succes")
    # batched_data = next(iter(loader))
    # data = dataset[1]
    # save_dataloader_iteration_as_video(data, "test.mp4", fps=7)