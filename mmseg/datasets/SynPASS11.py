# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support crop_pseudo_margins

# import os.path as osp
# import tempfile
# import mmcv
import numpy as np
from mmcv.utils import print_log#,get_root_logger
# from PIL import Image
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset

import os
import glob

@DATASETS.register_module()
class SynPASS_11_sun(CustomDataset):
    
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                'sky', 'person', 'car', 'private')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 crop_pseudo_margins=None,
                 img_suffix='.png',
                 seg_map_suffix='_trainID_13.png', # modified
                 **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super(SynPASS_11_sun, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [1024, 2048]

    def pre_pipeline(self, results):
        super(SynPASS_11_sun, self).pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load SynPASS-style annotations from directory.

        Args:
            folder (str): Root dataset folder, e.g., 'datasets/SynPASS'
            img_suffix (str): Suffix of images. Default: '.jpg'
            seg_map_suffix (str): Suffix of segmentation maps. Default: '_trainID.png'
            split (str): Dataset split. Default: 'train'
            weather (str): Weather condition filter. One of
                ['all', '/cloud', '/fog', '/rain', '/sun', 'day', 'night'].
                Default: 'all'.

        Returns:
            list[dict]: All image info of dataset.
        """
        folder = os.path.dirname(img_dir)
        img_suffix='.jpg'
        # seg_map_suffix='_trainID_13.png'
        split='train'
        weather='/sun'
        # 
        img_paths = glob.glob(os.path.join(folder, 'img', '*', split, '*', '*' + img_suffix))
        mask_paths = glob.glob(os.path.join(folder, 'semantic', '*', split, '*', '*' + seg_map_suffix))


        assert len(img_paths) == len(mask_paths), \
            f"Number mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"

        #  weather 
        if weather in ['/cloud', '/fog', '/rain', '/sun']:
            img_paths = [m for m in img_paths if weather in m]
            mask_paths = [m for m in mask_paths if weather in m]

        #  day/night 
        if weather in ['day', 'night']:
            new_img_paths, new_mask_paths = [], []
            list_file = os.path.join(folder, f'all_map_{weather}_val.txt')
            if not os.path.exists(list_file):
                raise FileNotFoundError(f"{list_file} not found for weather={weather}")

            with open(list_file, 'r') as f:
                all_map_dn_val = set([line.strip() for line in f])

            for img, mask in zip(img_paths, mask_paths):
                parts = img.split('/')
                if len(parts) >= 6:
                    p = parts[5]  #  SynPASS ：img/cloud/train/MAP_1_point2/000000.jpg
                    if p in all_map_dn_val:
                        new_img_paths.append(img)
                        new_mask_paths.append(mask)
            img_paths, mask_paths = new_img_paths, new_mask_paths

        # 
        img_paths = sorted(img_paths)
        mask_paths = sorted(mask_paths)
        assert len(img_paths) == len(mask_paths), \
            f"After filtering, mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"

        #  img_infos
        img_infos = []
        for img, mask in zip(img_paths, mask_paths):
            img_infos.append(dict(filename=img, ann=dict(seg_map=mask)))

        print_log(
            f'Loaded {len(img_infos)} images from {folder}, split={split}, weather={weather}',
            logger=get_root_logger())
        return img_infos
