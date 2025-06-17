import os
import json
import copy
import re

annotation_file = 'json_files/points_positions.json'
output_json = 'json_files/copied_annotations.json'

image_folder = 'model\data\images'

# Load base annotations
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Add video_id to each base annotation
for i, video_id in enumerate(['video_1', 'video_2', 'video_3', 'video_4']):
    if i < len(data):
        data[i]['data']['video_id'] = video_id

# Save modified version
with open('json_files/points_positions_with_ids.json', 'w') as f:
    json.dump(data, f, indent=2)

# Build a mapping from video_id to annotation
video_to_annotation = {
    ann['data']['video_id']: ann
    for ann in data if 'video_id' in ann['data']
}

all_new_tasks = []

# List all image files
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for img_file in image_files:
    match = re.match(r'(video_\d+)_frame_\d+\.jpg', img_file)
    if not match:
        print(f"⚠️ Skipping file with unexpected name: {img_file}")
        continue

    video_id = match.group(1)
    annotation = video_to_annotation.get(video_id)

    if annotation is None:
        print(f"⚠️ No annotation found for {video_id}")
        continue

    # Create annotation task
    new_task = copy.deepcopy(annotation)
    new_task['id'] = None
    new_task['data'] = copy.deepcopy(annotation['data'])
    new_task['data']['image'] = f'/data/upload/{img_file}'
    new_task['annotations'] = [{
        "result": annotation['annotations'][0]['result']
    }]
    all_new_tasks.append(new_task)

# Save all new annotation tasks
with open(output_json, 'w') as f:
    json.dump(all_new_tasks, f, indent=2)

print(f"\n✅ Prepared {len(all_new_tasks)} annotation tasks. Saved to {output_json}")
