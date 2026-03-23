# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add valid_mask_size
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset
import os.path as osp
import os
import mmcv
@DATASETS.register_module()
class ACDCDataset_13(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset_13, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds_13.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]
    
    CLASSES = (
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'car', 'private')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """
        img_dir: RGB  /path/to/ACDC/rgb_anon
        ann_dir: GT  /path/to/ACDC/gt
        split: train/val/test
        """


        split_name = split.split('/')[-1]  # train/val/test

        img_infos = []

        for weather_dir in sorted(os.listdir(img_dir)):
            weather_img_dir = osp.join(img_dir, weather_dir, split_name)
            if not osp.exists(weather_img_dir):
                continue

            weather_ann_dir = None
            if ann_dir is not None:
                weather_ann_dir = osp.join(ann_dir, weather_dir, split_name)
                if not osp.exists(weather_ann_dir):
                    weather_ann_dir = None

            for root, _, files in os.walk(weather_img_dir):
                for file in files:
                    if not file.endswith(img_suffix):
                        continue

                    img_path = osp.relpath(osp.join(root, file), img_dir)
                    img_info = dict(filename=img_path)

                    if weather_ann_dir is not None:
                        relative_path = osp.relpath(root, weather_img_dir)
                        label_file = file.replace(img_suffix, seg_map_suffix)
                        seg_map = osp.join(weather_ann_dir, relative_path, label_file)
                        if not osp.exists(seg_map):
                            raise FileNotFoundError(f"Seg map not found: {seg_map}")
                        img_info['ann'] = dict(seg_map=seg_map)

                    img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} images from {img_dir}, split={split_name}')
        return img_infos
