#summary 
#1 Load Configuration:
#
#	Reads config.yaml to get settings for:
#
#		Image transformations (resize, tensor conversion, normalization)
#		Dataset location and parameters
#		Dataloader settings
#		Augmentation and frame-related options
#
#2 Build Transformations:
#
#   Applies resizing and ToTensor() if specified.
#
#3 Initialize Dataset:
#
#   Uses HDF5ImageDataset to load images from .h5 files.
#   Applies transformations and optional normalization/augmentation.
#
#4 Create DataLoader:
#
#   Loads batches of images with specified batch_size and num_workers.
#
#5 Visualize First Batch:
#
#   Converts images from [-1, 1] to [0, 1] range.
#   Displays up to 8 images using matplotlib.
#
#
#
#6 Close Dataset:
#
#   Ensures HDF5 files are properly closed.


# Import  libraries
import os
import h5py  # For reading HDF5 files
import torch
from torch.utils.data import Dataset, DataLoader  # For PyTorch dataset and data loading
from torchvision import transforms  # For image transformations
from PIL import Image  # For image handling
import numpy as np
import yaml  # For reading configuration files
import random
import glob  # For file pattern matching
import matplotlib.pyplot as plt  # For visualization

# Disable HDF5 file locking (useful in some environments like NFS)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class HDF5ImageDataset(Dataset):
    def __init__(self, h5_folder, dataset_names=None, transform=None, normalize=True, aug=None, size=None, scale_min=0.08, scale_max=1.0, num_frames=1, frame_rate=1):
        """
        Custom PyTorch Dataset for loading images from HDF5 files.

        Args:
            h5_folder (str): Path to folder containing HDF5 files.
            dataset_names (list of str or None): Specific datasets (keys) to use from HDF5 files.
            transform (callable or None): Transformations to apply to images.
            normalize (bool): Whether to normalize images to [-1, 1].
            aug (str or None): Augmentation type.
            size (int or None): Target size for cropping/resizing.
            scale_min (float): Minimum scale for random crop.
            scale_max (float): Maximum scale for random crop.
            num_frames (int): Number of frames to sample per item.
            frame_rate (int): Interval between frames.
        """
        # Find all HDF5 files in the folder
        self.hdf5_paths = sorted(
            glob.glob(os.path.join(h5_folder, "*.hdf5")) + glob.glob(os.path.join(h5_folder, "*.h5"))
        )
        if not self.hdf5_paths:
            raise RuntimeError(f"No HDF5 files found in folder: {h5_folder}")

        # Ensure dataset_names is a list
        if dataset_names is not None and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names

        # Open all HDF5 files and collect dataset keys and lengths
        self.files = [h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*500, rdcc_nslots=375000) for path in self.hdf5_paths]
        self.lengths = []
        self.file_keys = []
        for file in self.files:
            keys = list(file.keys())
            if self.dataset_names is not None:
                keys = [k for k in keys if k in self.dataset_names]
            self.file_keys.append(keys)
            self.lengths.append({key: len(file[key]) for key in keys}) # type: ignore[attr-defined]

        # Total number of samples across all files
        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)

        # Set up augmentation and transformation
        self.aug = aug
        if self.aug == 'resize_center' and size is not None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
            ])
        elif self.aug == 'random_resize_center' and size is not None:
            self.custom_crop = RandomResizedCenterCrop(size=size, scale=(scale_min, scale_max))
            self.transform = transforms.Compose([
                self.custom_crop,
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.normalize = normalize
        self.num_frames = num_frames
        self.frame_rate = frame_rate

    def __len__(self):
        return self.total_length

    def apply_same_transform_to_all(self, frames, transform):
        # Apply the same transformation to all frames
        if self.transform is not None:
            return torch.stack([transform(frame) for frame in frames], dim=0)
        else:
            return frames

    def __getitem__(self, idx):
        # Randomly select a file and dataset key
        file_index = random.randint(0, len(self.files) - 1)
        h5_file = self.files[file_index]

        key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
        key = self.file_keys[file_index][key_index]

        # Randomly select a starting index for frame sampling
        max_start = self.lengths[file_index][key] - (self.num_frames+1)*self.frame_rate
        img_start_index = random.randint(0, max_start)

        # Load frames and convert to PIL images
        images = [Image.fromarray(h5_file[key][img_start_index+i]) for i in range(self.num_frames)] # type: ignore[attr-defined]

        # Reset crop parameters if using random crop
        if self.aug == 'random_resize_center':
            self.custom_crop.reset()

        # Apply transformations
        images = self.apply_same_transform_to_all(images, self.transform)

        # Normalize to [-1, 1] if required
        return {
            'images': images*2 -1
        }

    def close(self):
        # Close all HDF5 files
        for file in self.files:
            file.close()

class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR): # type: ignore[attr-defined]
        self.scale = scale
        self.interpolation = interpolation
        self.size = size
        self.fixed_params = None

    def get_params(self, img):
        # Compute crop parameters only once per batch
        if self.fixed_params is None:
            width, height = img.size
            area = height * width
            aspect_ratio = width / height

            target_area = random.uniform(*self.scale) * area

            new_width = int(round((target_area * aspect_ratio) ** 0.5))
            new_height = int(round((target_area / aspect_ratio) ** 0.5))
            x1 = (new_width - self.size) // 2
            y1 = (new_height - self.size) // 2
            self.fixed_params = (new_width, new_height, x1, y1)
        return self.fixed_params    

    def __call__(self, img):
        # Apply resize and crop
        new_width, new_height, x1, y1 = self.get_params(img)
        img = img.resize((new_width, new_height), self.interpolation)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))

    def reset(self):
        # Reset crop parameters for next batch
        self.fixed_params = None

def build_transform(tf_cfg):
    # Build transformation pipeline from config
    tf_list = []
    if "resize" in tf_cfg and tf_cfg["resize"]:
        tf_list.append(transforms.Resize(tuple(tf_cfg["resize"])))
    if tf_cfg.get("to_tensor", True):
        tf_list.append(transforms.ToTensor())
    return transforms.Compose(tf_list) if tf_list else None

if __name__ == "__main__":
    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        assert isinstance(config, dict)

    # Build transformations from config
    tf_cfg = config.get("transforms", {})
    transform = build_transform(tf_cfg)
    normalize = tf_cfg.get("normalize", True)

    # Extract dataset and dataloader parameters from config
    h5_folder = config["input"]["h5_folder"]
    dataset_names = config["input"].get("dataset_names", None)
    batch_size = config["dataloader"].get("batch_size", 2)
    num_workers = config["dataloader"].get("num_workers", 0)
    aug = config.get("augmentation", None)
    size = config.get("size", None)
    scale_min = config.get("scale_min", 0.08)
    scale_max = config.get("scale_max", 1.0)
    num_frames = config.get("num_frames", 1)
    frame_rate = config.get("frame_rate", 1)

    # Create dataset and dataloader
    dataset = HDF5ImageDataset(
        h5_folder=h5_folder,
        dataset_names=dataset_names,
        transform=transform,
        normalize=normalize,
        aug=aug,
        size=size,
        scale_min=scale_min,
        scale_max=scale_max,
        num_frames=num_frames,
        frame_rate=frame_rate
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Visualize one batch of images
    print("Batch:")
    for batch_imgs in loader:
        imgs = batch_imgs['jpg']
        print("Batch imgs shape:", imgs.shape)
        # If multiple frames, visualize only the first frame
        if imgs.dim() == 5:  # [B, num_frames, C, H, W]
            imgs = imgs[:, 0]
        imgs = (imgs + 1) / 2  # Convert back to [0, 1] range
        imgs = imgs.cpu().numpy()
        #n = min(8, imgs.shape[0]) #only 8 images presented
        n = imgs.shape[0]
        fig, ax = plt.subplots(1,n, figsize=(n * 2, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            img = np.transpose(imgs[i], (1, 2, 0))
            ax[i].imshow(np.clip(img, 0, 1))
            ax[i].axis('off')
        fig.tight_layout()
        fig.savefig("example.jpg")
        plt.show()
        break

    # Close HDF5 files
    dataset.close()