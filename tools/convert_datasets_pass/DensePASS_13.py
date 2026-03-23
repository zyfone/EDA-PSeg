import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import mmcv

"""
old trainId → new trainId):
---------------------------------
 0 : road          → 0
 1 : sidewalk      → 1
 2 : building      → 2
 3 : wall          → 3
 4 : fence         → 4
 5 : pole          → 13   (private)
 6 : traffic light → 5
 7 : traffic sign  → 13   (private)
 8 : vegetation    → 6
 9 : terrain       → 7
10 : sky           → 8
11 : person        → 13   (private)
12 : rider         → 13   (private)
13 : car           → 9
14 : truck         → 13   (private)
15 : bus           → 10
16 : train         → 13   (private)
17 : motorcycle    → 11
18 : bicycle       → 12
255: ignore label  → 255
"""

# （map after trainId ）
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'traffic light', 'vegetation', 'terrain', 'sky',
    'car', 'bus', 'motorcycle', 'bicycle', 'private'
)

# cityscapes  19 （trainId ）
original_class_names = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

#  13（private ）
private_class = ["pole", "traffic sign", "person", "rider", "truck", "train"]

#  trainId →  trainId
id_map = {i: 255 for i in range(256)}  # ignore


keep_class_names = CLASSES[:-1]  # remmove 'private'
for new_id, name in enumerate(keep_class_names):
    if name in original_class_names:
        old_id = original_class_names.index(name)
        id_map[old_id] = new_id

#  private_class map to 13
for name in private_class:
    if name in original_class_names:
        old_id = original_class_names.index(name)
        id_map[old_id] = 13


def convert_to_13(label_path):
    label = np.array(Image.open(label_path), dtype=np.uint8)
    label_13 = np.full_like(label, 255, dtype=np.uint8)

    for old_id in np.unique(label):
        label_13[label == old_id] = id_map[old_id]

    out_path = label_path.replace('labelTrainIds.png', 'labelTrainIds_13.png')
    Image.fromarray(label_13).save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert labelTrainIds to labelTrainIds_13")
    parser.add_argument('input_dir', help='Directory with labelTrainIds.png files')
    parser.add_argument('--nproc', type=int, default=1, help='Number of processes')
    return parser.parse_args()


def main():
    args = parse_args()

    #  *_labelTrainIds.png 
    label_paths = [
        osp.join(args.input_dir, f)
        for f in mmcv.scandir(args.input_dir, suffix='labelTrainIds.png', recursive=True)
    ]

    print(f'Found {len(label_paths)} label files.')

    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_to_13, label_paths, nproc=args.nproc)
    else:
        mmcv.track_progress(convert_to_13, label_paths)


if __name__ == '__main__':
    main()
