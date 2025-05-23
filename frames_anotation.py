import os
import json
import cv2
import copy

annotation_file = 'json_files/points_positions.json'
output_json = 'json_files/copied_annotations.json'

base_video_path = 'processed_frames'
video_folders = [f'video_{i}' for i in range(1, 5)]

with open('json_files/points_positions.json', 'r') as f:
    data = json.load(f)
    
for i, video_id in enumerate(['video_1','video_2', 'video_3', 'video_4']):
    if i < len(data):
        data[i]['data']['video_id'] = video_id

with open('json_files/points_positions_with_ids.json', 'w') as f:
    json.dump(data, f, indent=2)

all_new_tasks = []

# Build a mapping from video_id to annotation
video_to_annotation = {
    ann['data'].get('video_id'): ann
    for ann in data if 'video_id' in ann['data']
}

for video_folder in video_folders:
    frame_dir = os.path.join(base_video_path, video_folder, 'undistorted')
    output_image_dir = os.path.join(base_video_path, video_folder, 'annotated')
    os.makedirs(output_image_dir, exist_ok=True)

    base = video_to_annotation.get(video_folder)
    if base is None:
        print(f"⚠️ No annotation found for {video_folder}")
        continue

    base_results = base['annotations'][0]['result']

    image_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for img_file in image_files:
        img_path = os.path.join(frame_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Couldn't read {img_file} in {video_folder}")
            continue

        height, width = img.shape[:2]

        new_task = copy.deepcopy(base)
        new_task['id'] = None
        new_task['data'] = copy.deepcopy(base['data'])
        new_task['data']['image'] = f'/data/upload/{img_file}'
        new_task['annotations'] = [{
            "result": base_results
        }]
        all_new_tasks.append(new_task)

        # Draw annotations
        for result in base_results:
            if result['type'] == 'keypointlabels':
                x_pct = result['value']['x']
                y_pct = result['value']['y']
                orig_w = result['original_width']
                orig_h = result['original_height']

                x_rel = x_pct / 100 * orig_w
                y_rel = y_pct / 100 * orig_h

                x = int(x_rel / orig_w * width)
                y = int(y_rel / orig_h * height)

                cv2.circle(img, (x, y), 12, (0, 165, 255), -1)

        annotated_path = os.path.join(output_image_dir, img_file)
        cv2.imwrite(annotated_path, img)

    print(f"✅ Processed {len(image_files)} frames for {video_folder}")

print(f"\n✅ All annotations completed")

with open(output_json, 'w') as f:
    json.dump(all_new_tasks, f, indent=2)

print(f"✅Copied annotations to {len(all_new_tasks)} frames. Saved to {output_json}.")