import cv2
import os
import numpy as np

input_folder = 'videos'
output_folder = 'frames_extracted'
dataset_root = "model/data"   
n = 500  # Number of evenly spaced frames to extract

os.makedirs(output_folder, exist_ok=True)

dataset_imgs_dir = os.path.join(dataset_root, "images")
os.makedirs(dataset_imgs_dir, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):
        video_path = os.path.join(input_folder, filename)
        video_name = os.path.splitext(filename)[0]
        frame_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(frame_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        selected_indices = np.linspace(0, total_frames - 1, n, dtype=int)

        frame_count = 0
        saved_count = 0
        selected_set = set(selected_indices)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count in selected_set:
                fname = f'frame_{frame_count:05d}.jpg'
                frame_path = os.path.join(frame_output_dir, fname)
                cv2.imwrite(frame_path, frame)
                cv2.imwrite(os.path.join(dataset_imgs_dir, f'{video_name}_{fname}'), frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f'Extracted {saved_count} frames from {filename}')
