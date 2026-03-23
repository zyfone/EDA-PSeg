import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import mmcv
import functools


"""
    Build a mapping from original Cityscapes 19-class IDs (trainIds)
    to a reduced 13-class (or 13+1-class) label space for ACDC dataset.

    The mapping logic:
      - For 'train' split:
          * Keep only the 13 main classes listed in CLASSES[:-1]
            ('road', 'sidewalk', ..., 'car')
          * All other classes (e.g., 'rider', 'truck', 'bus', etc.) 
            are mapped to 255 (ignored label)
      - For 'val' / 'test' split:
          * Keep the same 13 main classes as above
          * Map all other Cityscapes classes to class index 13 ('private')

    Mapping reference table:
    ┌───────────┬────────────────────┬───────────────────────┬──────────────────────┬──────────────┬──────────────┐
    │ OriginalID│ Original ClassName │ Target Class Index    │ Target Class Name    │ train_split  │ val/test_split │
    ├───────────┼────────────────────┼───────────────────────┼──────────────────────┼──────────────┼──────────────┤
    │ 0         │ road               │ 0                     │ road                 │ 0            │ 0            │
    │ 1         │ sidewalk           │ 1                     │ sidewalk             │ 1            │ 1            │
    │ 2         │ building           │ 2                     │ building             │ 2            │ 2            │
    │ 3         │ wall               │ 3                     │ wall                 │ 3            │ 3            │
    │ 4         │ fence              │ 4                     │ fence                │ 4            │ 4            │
    │ 5         │ pole               │ 5                     │ pole                 │ 5            │ 5            │
    │ 6         │ traffic light      │ 6                     │ traffic light        │ 6            │ 6            │
    │ 7         │ traffic sign       │ 7                     │ traffic sign         │ 7            │ 7            │
    │ 8         │ vegetation         │ 8                     │ vegetation           │ 8            │ 8            │
    │ 9         │ terrain            │ 9                     │ terrain              │ 9            │ 9            │
    │ 10        │ sky                │ 10                    │ sky                  │ 10           │ 10           │
    │ 11        │ person             │ 11                    │ person               │ 11           │ 11           │
    │ 12        │ rider              │ —                     │ (ignored/private)    │ 255          │ 13           │
    │ 13        │ car                │ 12                    │ car                  │ 12           │ 12           │
    │ 14        │ truck              │ —                     │ (ignored/private)    │ 255          │ 13           │
    │ 15        │ bus                │ —                     │ (ignored/private)    │ 255          │ 13           │
    │ 16        │ train              │ —                     │ (ignored/private)    │ 255          │ 13           │
    │ 17        │ motorcycle         │ —                     │ (ignored/private)    │ 255          │ 13           │
    │ 18        │ bicycle            │ —                     │ (ignored/private)    │ 255          │ 13           │
    └───────────┴────────────────────┴───────────────────────┴──────────────────────┴──────────────┴──────────────┘
    """

# ACDC 13-class/13+1-class labelTrainIds
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'car', 'private'
)

# Cityscapes raw (trainId)
original_class_names = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]


def build_id_map(split="train"):
    id_map = {i: 255 for i in range(256)}
    for old_id, name in enumerate(original_class_names):
        if name in CLASSES[:-1]:  #  0~12
            id_map[old_id] = CLASSES.index(name)
        else:
            id_map[old_id] = 13 if split in ["val", "test"] else 255
    return id_map


def convert_worker(label_path, id_map):
    label = np.array(Image.open(label_path), dtype=np.uint8)
    label_13 = np.full_like(label, 255, dtype=np.uint8)
    for old_id in np.unique(label):
        label_13[label == old_id] = id_map.get(old_id, 255)
    out_path = label_path.replace('_gt_labelTrainIds.png', '_gt_labelTrainIds_13.png')
    Image.fromarray(label_13).save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ACDC gt_labelIds to 13-class/13+1-class labelTrainIds")
    parser.add_argument('base_dir', help='ACDC dataset base directory, e.g. /path/to/ACDC')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--nproc', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    gt_root = osp.join(args.base_dir, 'gt')
    if not osp.exists(gt_root):
        raise FileNotFoundError(f"gt directory does not exist: {gt_root}")


    id_map = build_id_map(split=args.split)
    label_paths = []


    for weather_dir in os.listdir(gt_root):
        weather_path = osp.join(gt_root, weather_dir)
        if not osp.isdir(weather_path):
            continue
        split_path = osp.join(weather_path, args.split)
        if not osp.exists(split_path):
            continue
        # find *_gt_labelIds.png 
        for root, _, files in os.walk(split_path):
            for file in files:
                if file.endswith('_gt_labelTrainIds.png'):
                    label_paths.append(osp.join(root, file))

    print(f'Found {len(label_paths)} label files under {gt_root}, split={args.split}.')
    if not label_paths:
        print("No label files found. Please check your dataset structure.")
        return

    worker_func = functools.partial(convert_worker, id_map=id_map)
    if args.nproc > 1:
        mmcv.track_parallel_progress(worker_func, label_paths, nproc=args.nproc)
    else:
        mmcv.track_progress(worker_func, label_paths)

# ----------------------------
if __name__ == '__main__':
    main()
