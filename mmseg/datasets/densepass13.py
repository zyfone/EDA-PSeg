# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset_13
from .builder import DATASETS
from .custom import CustomDataset

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
from mmseg.utils import get_root_logger

@DATASETS.register_module()
class DensePASSDataset_13(CustomDataset):
    CLASSES = CityscapesDataset_13.CLASSES
    PALETTE = CityscapesDataset_13.PALETTE

    def __init__(self, **kwargs):
        # assert kwargs.get('split') in [None, 'train']
        # if 'split' in kwargs:
        #     kwargs.pop('split')
        super(DensePASSDataset_13, self).__init__(
            img_suffix='.png',
            # seg_map_suffix='labelTrainIds.png',
            seg_map_suffix='labelTrainIds_13.png',
            split=None,
            **kwargs)
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory.
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded.

        Returns:
            list[dict]: All image info of dataset.
        """

        # print("using my load_annotations@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(img_dir,seg_map_suffix)
        img_infos = []

        for img in mmcv.scandir(img_dir, recursive=True):
            img_info = dict(filename=img)

            if ann_dir is not None:
                
                img_stem = osp.splitext(img)[0]         # 
                base_name = img_stem.rstrip('_')        #
                
                seg_map = f'{base_name}_{seg_map_suffix}'  # 
                # seg_map = f'{base_name}_labelTrainIds.png' 
                img_info['ann'] = dict(seg_map=seg_map)

            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos
