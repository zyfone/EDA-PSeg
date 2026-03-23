# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation, ignore class 13, validate labels

# [  class name, trainId ]
#     ('road', 0),
#     ('sidewalk', 1),
#     ('building', 2),
#     ('wall', 3),
#     ('fence', 4),
#     ('traffic light', 5),
#     ('vegetation', 6),
#     ('terrain', 7),
#     ('sky', 8),
#     ('car', 9),
#     ('bus', 10),
#     ('motorcycle', 11),
#     ('bicycle', 12),
#     ('pole', 13),
#     ('traffic sign', 13),
#     ('person', 13),
#     ('rider', 13),
#     ('truck', 13),
#     ('train', 13)
# ]


# ---------------------------------------------------------------------------- #
# Cityscapes 13-Class Mapping

# trainId : class_name
#  0  -> road
#  1  -> sidewalk
#  2  -> building
#  3  -> wall
#  4  -> fence
#  5  -> traffic light
#  6  -> vegetation
#  7  -> terrain
#  8  -> sky
#  9  -> car
# 10  -> bus
# 11  -> motorcycle
# 12  -> bicycle
#
# trainId = 255 means "ignore label"
# (includes person, rider, pole, truck, train, traffic sign, etc.)
# ---------------------------------------------------------------------------- #

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts13.preparation.json2labelImg13 import json2labelImg
from PIL import Image

def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds_13.png')
    json2labelImg(json_file, label_file, 'trainIds')

    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label).copy()  

        # 替换类别13为255（ignore）
        if np.any(label == 13):
            print(f"[Info] Found class 13 in {label_file}, converting to 255.")
            label[label == 13] = 255

       
        unique_labels = np.unique(label)
        invalid_labels = [int(l) for l in unique_labels if l not in range(13) and l != 255]
        if invalid_labels:
            print(f"[Warning] Invalid label(s) in {label_file}: {invalid_labels}")

        Image.fromarray(label.astype(np.uint8), mode='L').save(label_file)

        sample_class_stats = {}
        for c in range(13):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[c] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='Cityscapes dataset root')
    parser.add_argument('--gt-dir', default='gtFine', type=str, help='Subdir with annotations')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument('--nproc', default=8, type=int, help='Number of processes')
    return parser.parse_args()


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats_13.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict_13.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class_13.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path

    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats_13.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

if __name__ == '__main__':
    main()
