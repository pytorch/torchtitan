# nxtaim_dataloaders

This repository contains code to load TLD, ECP and atCity datasets. Data-loading requirements:
1. Select all videos and their corresponding metadata from a given directory.
2. Load contiguous sub-array of frames (and metadata) from a video.
3. Pre-determined format for the data directory.

### Traffic Light Dataset

#### Directory Structure:
```bash
TLD
├── Duesseldorf
│   └── Duesseldorf1
│       └── rec_2015-04-22-14-47-36
│           ├── cam_ampel_left_image_raw.mp4
│           ├── cam_ampel_right_image_raw.mp4
│           ├── cam_stereo_left_image_raw.mp4
│           ├── cam_stereo_right_image_raw.mp4
│           └── metadata.json
├── Koeln
│   └── Koeln4
│       ├── rec_2015-04-23-14-30-50
...
```

#### Usage:
- Use class TrafficLightDataset to get a PyTorch dataset. Pass in appropriate arguments and use this dataset to pass into a dataloader.
- Example usage:
    ```python
    dataset_root = "TLD"
    # transforms = ... # torch transforms, e.g.: resize images to a consistent shape
    dataset = TrafficLightDataset(dataset_root, video_type='ampel', frame_interval=10, frame_step=2) # add transform=transforms, if required
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    print(next(iter(loader)))
    ```

### ECP Dataset

#### Directory Structure:
```bash
ECP
├── Euro-cityAutobahn-scenarioPolice_2017-04-30-12-44-31
│   ├── cam_junction_left_image_raw.mp4
│   ├── cam_junction_right_image_raw.mp4
│   ├── cam_stereo_ar0230_left_image_raw.mp4
│   ├── cam_stereo_ar0230_right_image_raw.mp4
│   ├── metadata_flexray_full.json
│   ├── metadata_flexray_synced.json
│   ├── timestamps.npy
│   └── timestamps.txt
...
```

#### Usage:
- Note: ECP Dataloader is usable for both ECP and At_City dataset
- Use class ECPDataset to get a PyTorch dataset. Pass in appropriate arguments and use this dataset to pass into a dataloader.
- Example usage:
    ```python
    dataset_root = "ECP"
    # transforms = ... # torch transforms, e.g.: resize images to a consistent shape
    dataset = ECPDataset(dataset_root, frame_interval=10, frame_step=2) # add transform=transforms, if required
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    print(next(iter(loader)))


### Adverse Weather Dataset
#### Directory Structure:
aw_data
<pre>
├─ videos
│   ├── 2018-02-06_17-54-34.mp4
│   ├── 2018-02-06_17-55-51.mp4
│   ├── 2018-02-06_17-56-23.mp4
├─ pointclouds
│   └── 2018-12-11_14-22-18
│       ├── 1544534541407839835.npy
│       ├── 1544534541504522285.npy
│       ├── 1544534541620296446.npy
│       ├── ...
│   └── 2018-09-03_19-16-24
│       ├── ...
...


</pre>


### Usage Video Dataloader:
- Use AWDataset to get a PyTorch dataset. Pass in appropriate arguments and use this dataset to pass into a dataloader.
- Example Usage:
split_file = 'aw_data/video_split.txt'
    idx_map = read_split_file(split_file)
    dataset_root = "/p/data1/nxtaim/proprietary/mb_ag/aw_data/videos"
    dataset = AWDataset(
        video_dir=dataset_root,
        idx_map=idx_map,
        frame_interval=50,
        frame_step=2,
        blur=True,
    )


### Usage Lidar Dataloader:
- Specify within the dataset root, interval of pointclouds to be sampled, and the number of points for each pointcloud to be loaded (this ensures each pointcloud consists of the same number of points).
- Example Usage:
dataset_root = "awdata/pointclouds"
    dataset = AWLidarDataset(
        lidar_dir=dataset_root,
        interval=50,
        step=5,
        points=55000,
    )

