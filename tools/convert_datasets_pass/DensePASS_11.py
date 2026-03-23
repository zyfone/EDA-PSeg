import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import mmcv
from functools import partial

# =============================
#
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'car'
)

# =============================
original_class_names = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]


private_class = ['bus', 'truck', 'train', 'motorcycle', 'bicycle', 'rider']


id_map = {i: 255 for i in range(256)} 


for new_id, name in enumerate(CLASSES):
    old_id = original_class_names.index(name)
    id_map[old_id] = new_id

for name in private_class:
    old_id = original_class_names.index(name)
    id_map[old_id] = 13

"""
===  (old_id → new_id) ===
 0 road         → 0
 1 sidewalk     → 1
 2 building     → 2
 3 wall         → 3
 4 fence        → 4
 5 pole         → 5
 6 traffic light→ 6
 7 traffic sign → 7
 8 vegetation   → 8
 9 terrain      → 9
10 sky          → 10
11 person       → 11
12 rider        → 13 (private)
13 car          → 12
14 truck        → 13 (private)
15 bus          → 13 (private)
16 train        → 13 (private)
17 motorcycle   → 13 (private)
18 bicycle      → 13 (private)
 (19~255)   → 255 (ignore)
"""

def convert_to_13(label_path):
    label = np.array(Image.open(label_path), dtype=np.uint8)
    label_13 = np.full_like(label, 255, dtype=np.uint8)

    for old_id in np.unique(label):
        label_13[label == old_id] = id_map[old_id]

    base, ext = osp.splitext(label_path)
    out_path = base + "_syn_13" + ext
    Image.fromarray(label_13, mode="L").save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes labelTrainIds to 13-class labels"
    )
    parser.add_argument("input_dir", help="Directory with labelTrainIds.png files")
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of processes")
    return parser.parse_args()


def main():
    args = parse_args()

    # 查找所有 *_labelTrainIds.png 文件
    label_paths = [
        osp.join(args.input_dir, f)
        for f in mmcv.scandir(args.input_dir, suffix='labelTrainIds.png', recursive=True)
    ]

    print(f"Found {len(label_paths)} label files.")

    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_to_13, label_paths, nproc=args.nproc)
    else:
        mmcv.track_progress(convert_to_13, label_paths)


if __name__ == '__main__':
    main()
