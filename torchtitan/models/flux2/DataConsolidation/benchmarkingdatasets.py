import os, sys
from collections.abc import Mapping
import numpy as np
import time
from tqdm.auto import tqdm

from omegaconf import OmegaConf

from torch import Tensor

basepath = os.path.dirname(__file__)
if not basepath in sys.path:
    sys.path.insert(0, basepath)
from src.multifiledataset import MultiFileDataset
from src.ffmpegvideowriter import FFMPEGVideoWriter
from src.utils import instantiate_from_config

def nested_dict_to_str(d: dict, prefix:str = "\n  - ", max_len=40):
    subprefix = prefix.replace('|', ' ').replace('-','|')+" - "
    ret = prefix + prefix.join([
        f"'{k}': "+
            (f"{v.shape} ({v.dtype})" if isinstance(v, Tensor)
            else nested_dict_to_str(v, prefix=subprefix) if isinstance(v, dict)
                else (str(v) if len(str(v))<=max_len else str(v)[:max_len]+"...")+f" ({type(v)})" )
            for k,v in d.items()
    ])
    return ret

multi_frame_possible_keys = [
    'images','camera_front_medium', 'camera_front_wide', 
    'camera_rear_wide', 'camera', 'frames'
]

class DatasetBenchmark:
    def __init__(self, datasets):
        assert isinstance(datasets, Mapping), r"need datasets as {name: dataset, ...}"
        self.datasets = datasets
    
    def benchmark(
        self, 
        samples_per_dataset=20,
        vid_save_path=None,
        fps=2
    ):  
        times = {name: 0.0 for name in self.datasets.keys()}
        wall_times = {name: 0.0 for name in self.datasets.keys()}
        frames = {name: 1 for name in self.datasets.keys()}
        pbar = tqdm(total=samples_per_dataset * len(self.datasets), 
                    desc="speed: "+", ".join([f"{name}= n/a  s/it" for name in self.datasets.keys()]),
                    dynamic_ncols=True)
        if vid_save_path is None:
            vid_save_path = os.path.dirname(os.path.dirname(__file__)) + "/benchmarking_video"
        with FFMPEGVideoWriter(
            video_path=vid_save_path,
            fps=fps,
            font_size=12,
            quality=28,
            preset="veryslow",
        ) as videowriter:
            updates = 0
            for j in range(samples_per_dataset):
                for name, dataset in self.datasets.items():
                    try:
                        idx = np.random.randint(0, len(dataset))
                    except ValueError as e:
                        print(f"Dataset '{name}' has a weird property len, namely: {len(dataset) if hasattr(dataset, '__len__') else 'no __len__ attribute'}")
                        raise e
                    if False: #isinstance(dataset, MultiFileDataset):
                        file_idx, sample_idx = dataset._global2local_idx(idx)
                        file_path = dataset._file_paths[file_idx]
                        metadata = dataset._file_metadatas[file_idx]
                        
                        start_time = time.process_time()
                        start_wall_time = time.time()
                        batch = dataset._getitem(
                            file_path=file_path,
                            sample_idx=sample_idx,
                            metadata=metadata,
                        )
                        process_time = time.process_time() - start_time
                        wall_time = time.time() - start_wall_time

                        img, title = dataset._get_image_representation(
                            file_path=file_path,
                            sample_idx=sample_idx,
                            metadata=metadata,
                            batch=batch
                        )
                        videowriter.write(img=img, title=f"{name}: {title}")
                    else:
                        start_time = time.process_time()
                        start_wall_time = time.time()
                        batch = dataset[idx]
                        process_time = time.process_time() - start_time
                        wall_time = time.time() - start_wall_time
                        single_frame_key = [k for k in ['jpg',] if k in batch]
                        multi__frame_key = [k for k in multi_frame_possible_keys if k in batch]
                        cnt = 0
                        if len(single_frame_key) > 0:
                            for k in single_frame_key:
                                img = (batch[k].cpu().numpy()*127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)
                                videowriter.write(img=img, title=f"{name}: idx {idx}_{cnt}")
                                cnt += 1
                        elif len(multi__frame_key) > 0:
                            for k in multi__frame_key:
                                vid_img = batch[k].cpu().numpy()
                                if vid_img.ndim > 4:
                                    vid_img = vid_img.reshape((-1, *vid_img.shape[-3:]))
                                if vid_img.ndim == 4:
                                    for vid_idx in range(vid_img.shape[0]):
                                        img = (vid_img[vid_idx]*127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)
                                        videowriter.write(img=img, title=f"{name}: idx {idx}_{cnt}")
                                        cnt += 1
                                else:
                                    img = (vid_img*127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)
                                    videowriter.write(img=img, title=f"{name}: idx {idx}_{cnt}")
                                    cnt += 1
                        else:
                            print(f"\nWarning! Dataset '{name}' uses non of the known keys for camera frames")
                    frames[name] = cnt
                    times[name] += (process_time - times[name]) / (j+1)
                    wall_times[name] += (wall_time - wall_times[name]) / (j+1)
                    updates += 1
                    if j == 0:
                        print(f"An item of dataset '{name}' contains:"+
                               nested_dict_to_str(d=batch), end="\n\n", flush=True)
                        #print(f"and images have properties: shape {batch[k].shape}, dtype {batch[k].dtype}, min {batch[k].min()}, max {batch[k].max()}\n")
                    else:
                        pbar.update(updates)
                        updates = 0
                pbar.set_description("speed: "+", ".join([f"{name}={times[name]:.3f} s/it" for name in self.datasets.keys()]))
            pbar.close()
        name_len = max([len(name) for name in self.datasets.keys()])
        print("\n - ".join(["final speeds: (note that processes might benefit from idle other CPUs during benchmarking for shorter wall time)"]+
                           [f"{name:>{name_len}}: {times[name]:6.3f} s/it ({wall_times[name]:6.3f} s walltime/it, {times[name]/frames[name]:6.3f} s/it/frame) for {frames[name]:2d} frames" for name in self.datasets.keys()]))

if __name__ == "__main__":
    
    config_name = sys.argv[1] if len(sys.argv) > 1 else "sample_datasets.yaml"
    yaml_path = os.path.dirname(__file__) + f"/config/{config_name}"
    cfg = OmegaConf.load(yaml_path)

    datasets = {}
    for dataset_cfg in cfg.datasets:
        name = list(dataset_cfg.keys())[0]
        start_time = time.process_time()
        dataset = instantiate_from_config(dataset_cfg[name], recursive=True, debug=False)
        process_time = time.process_time() - start_time
        datasets[name] = dataset
        print(f"Creating Dataset '{name}' took {process_time:.2f}s for {len(dataset)} samples") # type: ignore

    print("\nCreated Datasets\n")

    benchmark = DatasetBenchmark(
        datasets=datasets
    )
    benchmark.benchmark(samples_per_dataset=40)