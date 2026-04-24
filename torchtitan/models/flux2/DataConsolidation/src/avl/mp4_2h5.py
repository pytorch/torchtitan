import cv2
import h5py
import numpy as np
import os
 
def convert_video_to_hdf5(mp4_file_path, hdf5_file_path, frame_size):
    cap = cv2.VideoCapture(mp4_file_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
 
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        frames = []
 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, frame_size)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(resized_frame)
 
        frames_array = np.array(frames)
        hdf5_file.create_dataset('video_frames', data=frames_array, compression="gzip")
 
    cap.release()
    print(f"✅ Successfully converted '{mp4_file_path}' to '{hdf5_file_path}' with resized frames.")
 
# Ask user for filename and size
if __name__ == "__main__":
    filename = input("Enter the MP4 filename (in the same folder): ")
    width = int(input("Enter the desired frame width (e.g., 224): "))
    height = int(input("Enter the desired frame height (e.g., 224): "))
 
    mp4_path = os.path.join(os.getcwd(), filename)
    hdf5_filename = os.path.splitext(filename)[0] + ".h5"
    hdf5_path = os.path.join(os.getcwd(), hdf5_filename)
 
    convert_video_to_hdf5(mp4_path, hdf5_path, frame_size=(width, height))