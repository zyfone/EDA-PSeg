# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GTADataset_syn(CustomDataset):

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                'sky', 'person', 'car', 'private')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(GTADataset_syn, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds_11.png',
            split=None,
            **kwargs)
