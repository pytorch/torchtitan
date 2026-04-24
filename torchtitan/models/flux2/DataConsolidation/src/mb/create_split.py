import os
import cv2
from tqdm import tqdm



def creat_split_file(data_path, start_frame_idx=0, frame_step=2, frame_interval=50):
    video_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.mp4')]
    video_counter = 0
    idx_map = []
    frames_needed = frame_step * frame_interval

    for video_path in tqdm(video_paths):
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        if start_frame_idx + frames_needed <= frame_count:
            video_counter += 1

        for start_frame in range(start_frame_idx, frame_count, frames_needed):
            if start_frame + frames_needed <= frame_count:
                idx_map.append((video_path, start_frame))

    return idx_map, video_counter






if __name__ == "__main__":
    output_path =  '/home/alzuber/external/ext0/split.txt'
    data_path = '/p/data1/nxtaim/proprietary/mb_ag/aw_data/videos'
    start_frame_idx = 0
    frame_step = 2
    frame_interval = 50

    idx_map, video_counter = creat_split_file(data_path, start_frame_idx, frame_step, frame_interval)
    with open(output_path, 'w+') as f:
        for idx in idx_map:
            f.write(f"{idx[0]},{idx[1]}\n")

    print(f"Found {len([os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.mp4')])} over all. Usable videos found {video_counter}. Assorted {len([os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.mp4')]) - video_counter}")