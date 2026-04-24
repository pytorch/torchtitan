import argparse
import io
import json
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
from decord import VideoReader, cpu
from natsort import natsorted
from PIL import Image
from tqdm import tqdm


def stream_frames(
    video_path: Path, frames_list: Iterable[str]
) -> Iterable[Tuple[str, np.ndarray]]:
    """
    Input:
        video_path: path to video
        frame_ids: List of frame_names to be extracted from the video
    Yields:
        (frame_id, frame) one-by-one
    """
    frame_ids = [int(id.split("_")[-1]) for id in frames_list]

    vr = VideoReader(str(video_path), ctx=cpu(0))

    # Ensure requested frame IDs are in range
    max_id = len(vr) - 1
    frame_ids = [i for i in frame_ids if 0 <= i <= max_id]

    for frame_name, fid in zip(frames_list, frame_ids):
        frame = vr[fid].asnumpy()
        yield frame_name, frame


def encode_png_bytes(frame_np: np.ndarray, png_opts: dict | None = None) -> np.ndarray:
    img = Image.fromarray(frame_np, mode="RGB")
    buffer = io.BytesIO()
    save_kwargs = dict(format="PNG", optimize=True, compress_level=9)
    if png_opts:
        save_kwargs.update(png_opts)
    img.save(buffer, **save_kwargs)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-json", type=Path, help="json file with captions")
    parser.add_argument(
        "--video-root", type=Path, help="root directory of all video files"
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        help="hdf5 file where extracted images/captions are stored",
    )

    args = parser.parse_args()

    video_files = sorted(args.video_root.glob("*.mp4"))
    video_id = args.captions_json.stem
    args.output_h5.mkdir(parents=True, exist_ok=True)
    h5_output_file = Path(args.output_h5) / (args.captions_json.stem + ".h5")

    front_video = [
        vid for vid in video_files if (video_id in vid.name and "front" in vid.name)
    ][0]
    rear_video = [
        vid for vid in video_files if (video_id in vid.name and "rear" in vid.name)
    ][0]

    with open(args.captions_json, "r") as fh:
        captions = json.load(fh)

    frames_list_front = natsorted([k for k in captions.keys() if "front" in k])
    frames_list_rear = natsorted([k for k in captions.keys() if "rear" in k])

    total_len = len(frames_list_front) + len(frames_list_rear)

    with h5py.File(h5_output_file, "w") as hdf:
        dt_img = h5py.vlen_dtype(np.dtype(np.uint8))
        dt_str = h5py.string_dtype(encoding="utf-8")
        dset_image = hdf.create_dataset("image_bytes", shape=(total_len,), dtype=dt_img)
        dset_captions = hdf.create_dataset("captions", shape=(total_len,), dtype=dt_str)
        dset_filenames = hdf.create_dataset(
            "filenames", shape=(total_len,), dtype=dt_str
        )

        idx = 0
        print(f"")
        for frame_id, frame in tqdm(
            stream_frames(front_video, frames_list_front), total=len(frames_list_front)
        ):
            img_data = encode_png_bytes(frame)
            dset_image[idx] = img_data
            dset_captions[idx] = captions[frame_id]
            dset_filenames[idx] = frame_id

            idx += 1

        for frame_id, frame in tqdm(
            stream_frames(rear_video, frames_list_rear), total=len(frames_list_rear)
        ):
            img_data = encode_png_bytes(frame)
            dset_image[idx] = img_data
            dset_captions[idx] = captions[frame_id]
            dset_filenames[idx] = frame_id

            idx += 1

    print(f"Video frames extraction to {str(h5_output_file)} completed")


if __name__ == "__main__":
    main()
