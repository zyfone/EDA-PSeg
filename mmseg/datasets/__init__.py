# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .uda_dataset import UDADataset
from .densepass import DensePASSDataset
from .WildPASS2K import WildPASS2K
from .cityscapes13 import CityscapesDataset_13
from .densepass13 import DensePASSDataset_13
from .WildPASS2K_13 import WildPASS2K_13
from .SynPASS11 import SynPASS_11_sun
from .SynPASS11_weather import SynPASS_11_weather
from .gta_syn_11 import GTADataset_syn
from .acdc13 import ACDCDataset_13
from .WildPASS2K_11 import WildPASS2K_11
from .densepass11 import DensePASSDataset_11
__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    "DensePASSDataset",
    "WildPASS2K",
    "CityscapesDataset_13",
    "DensePASSDataset_13",
    "GTADataset_13",
    "SynPASS_11_sun",
    "SynPASS_11_weather",
    "WildPASS2K_11",
    "DensePASSDataset_11",
    "GTADataset_syn",
    "ACDCDataset_13",
    "WildPASS2K_13"
]
