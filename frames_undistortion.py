import os
import cv2
import numpy as np

input_root = "frames_extracted"            
output_root = "processed_frames"           
dataset_root = "model/data"                      

# === FISHEYE CAMERA PARAMS ===
def get_camera_params(width, height):
    K = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    D = np.array([-1, 0.9, 1.05, 0], dtype=np.float64)
    return K, D

for video_folder in sorted(os.listdir(input_root)):
    video_path = os.path.join(input_root, video_folder)
    if not os.path.isdir(video_path):
        continue

    frame_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Prepare output directories
    output_video_path = os.path.join(output_root, video_folder)
    original_dir = os.path.join(output_video_path, "original")
    undistorted_dir = os.path.join(output_video_path, "undistorted")
    dataset_imgs_dir = os.path.join(dataset_root, "images")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(undistorted_dir, exist_ok=True)
    os.makedirs(dataset_imgs_dir, exist_ok=True)

    for fname in frame_files:
        input_frame_path = os.path.join(video_path, fname)
        image = cv2.imread(input_frame_path)
        if image is None:
            continue
        
        image_cropped = image[8:, :]  # From 1520 → 1512 in height

        h, w = image_cropped.shape[:2]
        K, D = get_camera_params(w, h)

        # Undistort cropped image 
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        undistorted = cv2.remap(image_cropped, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imwrite(os.path.join(original_dir, fname), image_cropped) 
        cv2.imwrite(os.path.join(undistorted_dir, fname), undistorted)

        # Resize to 1280x720 before saving to dataset folder
        resized_for_dataset = cv2.resize(undistorted, (1280, 720))
        cv2.imwrite(os.path.join(dataset_imgs_dir, f'{video_folder}_{fname}'), resized_for_dataset)

    print(f"✅[{video_folder}] Processed {len(frame_files)} frames.")

print(f"\n✅ All undistortions completed")
