import argparse
import json
import os
import os.path as osp
import glob
import logging
import functools

import mmcv
import numpy as np
from PIL import Image



# -----------------------------------------------------------------------------
# SynPASS trainID -> shared_class mapping  (for training)
# -----------------------------------------------------------------------------
# trainID meaning in SynPASS:
#   0: Background
#   1: Building
#   2: Fence
#   3: Other
#   4: Pedestrian
#   5: Pole
#   6: RoadLine
#   7: Road
#   8: SideWalk
#   9: Vegetation
#   10: Vehicles
#   11: Wall
#   12: TrafficSign
#   13: Sky
#   14: Ground
#   15: Bridge
#   16: RailTrack
#   17: GroundRail
#   18: TrafficLight
#   19: Static
#   20: Dynamic
#   21: Water
#   22: Terrain
#
# shared_class (13 classes):
#   0:'road', 1:'sidewalk', 2:'building', 3:'wall', 4:'fence', 5:'pole',
#   6:'traffic light', 7:'traffic sign', 8:'vegetation', 9:'terrain',
#   10:'sky', 11:'person', 12:'car'
#
# Notes:
#   255 means "ignored class" (not used in training)
#   This mapping aligns SynPASS labels to the 13 shared semantic categories.
# -----------------------------------------------------------------------------

# trainid_to_shared_train = {
#     0: 255,   # Background       -> ignore
#     1: 2,     # Building         -> building
#     2: 4,     # Fence            -> fence
#     3: 255,   # Other            -> ignore (not included in shared classes)
#     4: 11,    # Pedestrian       -> person
#     5: 5,     # Pole             -> pole
#     6: 0,     # RoadLine         -> road
#     7: 0,     # Road             -> road
#     8: 1,     # SideWalk         -> sidewalk
#     9: 8,     # Vegetation       -> vegetation
#     10: 12,   # Vehicles         -> car
#     11: 3,    # Wall             -> wall
#     12: 7,    # TrafficSign      -> traffic sign
#     13: 10,   # Sky              -> sky
#     14: 255,  # Ground           -> ignore
#     15: 255,  # Bridge           -> ignore
#     16: 255,  # RailTrack        -> ignore
#     17: 255,  # GroundRail       -> ignore
#     18: 6,    # TrafficLight     -> traffic light
#     19: 255,  # Static           -> ignore
#     20: 255,  # Dynamic          -> ignore
#     21: 255,  # Water            -> ignore
#     22: 9,    # Terrain          -> terrain
# }


# ----------------------------
# Shared class
# ----------------------------
shared_class = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'car'
)

shared_class_test = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'car', 'private'
)


# SynPASS trainID -> shared_class 映射
trainid_to_shared_train = {
    0: 255,
    1: 2,
    2: 4,
    3: 255,
    4: 11,
    5: 5,
    6: 0,
    7: 0,
    8: 1,
    9: 8,
    10: 12,
    11: 3,
    12: 7,
    13: 10,
    14: 255,
    15: 255,
    16: 255,
    17: 255,
    18: 6,
    19: 255,
    20: 255,
    21: 255,
    22: 9,
}

trainid_to_shared_test = {
    0: 255,
    1: 2,
    2: 4,
    3: 13,  # private
    4: 11,
    5: 5,
    6: 0,
    7: 0,
    8: 1,
    9: 8,
    10: 12,
    11: 3,
    12: 7,
    13: 10,
    14: 13,
    15: 13,
    16: 13,
    17: 13,
    18: 6,
    19: 13,
    20: 13,
    21: 13,
    22: 9,
}


def read_list(file_path):
    with open(file_path) as f:
        return [line.rstrip() for line in f]


def _get_city_pairs(folder, splits=['train'], weather='all'):
    img_paths, mask_paths = [], []

    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(',') if s.strip()]

    for split in splits:
        img_paths.extend(
            glob.glob(osp.join(folder, 'img', '*', split, '*', '*.jpg'))
        )
        mask_paths.extend(
            glob.glob(osp.join(folder, 'semantic', '*', split, '*', '*_trainID.png'))
        )

    assert len(img_paths) == len(mask_paths), \
        f" {len(img_paths)} images vs {len(mask_paths)} masks"

    if weather is not None and weather != 'all':
        w = weather.lstrip('/')
        if w in {'cloud', 'fog', 'rain', 'sun'}:
            img_paths = [p for p in img_paths if f'/{w}/' in p]
            mask_paths = [p for p in mask_paths if f'/{w}/' in p]
        elif w in {'day', 'night'}:
            new_img_paths, new_mask_paths = [], []
            list_file = osp.join(folder, f'all_map_{w}_val.txt')
            allow = set(read_list(list_file)) if osp.exists(list_file) else set()
            for img, mask in zip(img_paths, mask_paths):
                p = osp.basename(osp.dirname(img))
                if p in allow:
                    new_img_paths.append(img)
                    new_mask_paths.append(mask)
            img_paths, mask_paths = new_img_paths, new_mask_paths

    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)
    assert len(img_paths) == len(mask_paths)

    logging.info(f'Found {len(img_paths)} images in {splits} of {folder}')
    return img_paths, mask_paths


def convert_and_stat(mask_file, mapping, convert=True, out_suffix='_trainID_13.png'):

    if not osp.exists(mask_file):
        raise FileNotFoundError(f"[Error] Label file not found: {mask_file}")

    label = np.array(Image.open(mask_file), dtype=np.uint8)
    label_mapped = np.full_like(label, 255, dtype=np.uint8)


    for old_id in np.unique(label):
        if int(old_id) in mapping:
            label_mapped[label == old_id] = mapping[int(old_id)]

    out_path = mask_file.replace('_trainID.png', out_suffix)
    if convert:
        Image.fromarray(label_mapped).save(out_path)
    used_file = out_path

    unique_labels, counts = np.unique(label_mapped, return_counts=True)
    sample_class_stats = {
        int(c): int(n) for c, n in zip(unique_labels, counts) if int(c) != 255
    }
    sample_class_stats['file'] = used_file
    return sample_class_stats




def worker(mask_file, mapping, convert=True):
    return convert_and_stat(mask_file, mapping=mapping, convert=convert)



def save_class_stats(out_dir, sample_class_stats, class_names, out_prefix='13'):
    sample_class_stats = [e for e in sample_class_stats if e is not None]


    with open(osp.join(out_dir, f'sample_class_stats_{out_prefix}.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)


    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        stats = dict(stats) 
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, f'sample_class_stats_dict_{out_prefix}.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)


    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            samples_with_class.setdefault(c, []).append((file, n))
    with open(osp.join(out_dir, f'samples_with_class_{out_prefix}.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


    global_class_stats = {}
    for stats in sample_class_stats_dict.values():
        for c, n in stats.items():
            global_class_stats[c] = global_class_stats.get(c, 0) + n

    total_pixels = int(sum(global_class_stats.values()))
    global_class_stats["total_pixels"] = total_pixels


    global_class_stats_named = {
        class_names[int(c)]: int(n) for c, n in global_class_stats.items() if c != "total_pixels"
    }
    global_class_stats_named["total_pixels"] = total_pixels

    print(f"[Info] Global class stats saved to {osp.join(out_dir, f'global_class_stats_{out_prefix}.json')}")
    print(f"[Info] Found {len(global_class_stats)-1} classes, total pixels = {total_pixels}")

    with open(osp.join(out_dir, f'global_class_stats_{out_prefix}.json'), 'w') as of:
        json.dump(global_class_stats_named, of, indent=2)



def parse_args():
    parser = argparse.ArgumentParser(description='Compute SynPASS class stats mapped to shared_class')
    parser.add_argument('synpass_path', help='SynPASS dataset root')
    parser.add_argument('--split', default='train', type=str,
                        help='Dataset split(s), comma separated, e.g. "train,val,test"')
    parser.add_argument('--weather', default='all', type=str, help='Weather condition filter: all/cloud/fog/rain/sun/day/night')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument('--nproc', default=8, type=int, help='Number of processes')
    parser.add_argument('--mapping', default='train', choices=['train', 'test'],
                        help='Use train or test mapping (test private class will be mapped to 13)')
    return parser.parse_args()


def main():
    args = parse_args()
    synpass_path = args.synpass_path
    out_dir = args.out_dir if args.out_dir else synpass_path
    mmcv.mkdir_or_exist(out_dir)

    if args.mapping == 'train':
        mapping = trainid_to_shared_train
        class_names = shared_class
    else:
        mapping = trainid_to_shared_test
        class_names = shared_class_test


    _, mask_paths = _get_city_pairs(synpass_path, splits=args.split, weather=args.weather)
    print(f'Found {len(mask_paths)} label files.')

    if args.nproc > 1:
        worker_partial = functools.partial(worker, mapping=mapping, convert=True)
        sample_class_stats = mmcv.track_parallel_progress(worker_partial, mask_paths, args.nproc)
    else:
        sample_class_stats = [worker(f, mapping=mapping, convert=True) for f in mask_paths]

    if args.mapping == 'train':
        save_class_stats(out_dir, sample_class_stats, class_names)


if __name__ == '__main__':
    main()
