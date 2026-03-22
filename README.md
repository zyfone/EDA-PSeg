<div align="center">
  <h2><b> Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation </b></h2>
</div>

<div align="center">
![](https://img.shields.io/github/stars/zyfone/EDA-PSeg?color=yellow)
![](https://img.shields.io/github/forks/zyfone/EDA-PSeg?color=lightblue)
![](https://img.shields.io/github/last-commit/zyfone/EDA-PSeg?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
</div>

<div align="center">
> [**Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation**](https://arxiv.org/abs/)<br>
> [Yuanfan Zheng<sup>1</sup>](https://github.com/zyfone/), [Kunyu Peng <sup>2</sup>](https://scholar.google.com/citations?user=pA9c0YsAAAAJ&hl=en), [Xu Zheng<sup>3</sup>](https://zhengxujosh.github.io/), [Kailun Yang*<sup>1</sup>](https://yangkailun.com/)<br> Hunan University<sup>1</sup>; IAR, Karlsruher Institut für Technologie<sup>2</sup>; Hong Kong University of Science and Technology (HKUST)<sup>3</sup>
</div>



## Data Preparation

download link:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [DensePASS](https://github.com/chma1024/DensePASS)
- [SynPASS](https://drive.google.com/file/d/1u-5J13CD6MXpWB53apB-L6kZ3hK1JR77/view?usp=sharing)
- [DensePASS](https://github.com/chma1024/DensePASS)
- [ACDC](https://acdc.vision.ee.ethz.ch/)
- [GTA](https://download.visinf.tu-darmstadt.de/data/from_games/)



**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell

#closed set PIN2PAN (Cityscapes，WildPASS2K → DensePASS).
python tools/convert_datasets_normal/cityscapes.py  XXXXXX  --nproc  8 #for source domain
python tools/convert_datasets_pass/target_empoty.py XXXXXXX  --nproc  8 #WildPASS2K for target domain
#Open-set PIN2PAN (Cityscapes，WildPASS2K → DensePASS).
python tools/convert_datasets_pass/cityscapes_13_train.py  XXXXXX  --nproc  8 #for source domain 
python tools/convert_datasets_pass/DensePASS_13.py  XXXXXX  --nproc  8 #for test
#Partial-set PIN2PAN (Cityscapes，WildPASS2K → DensePASS).

python tools/convert_datasets_pass/DensePASS_13_p.py  XXXXXX  --nproc  8  #for test

#Open Partial-set PIN2PAN (Cityscapes，WildPASS2K → DensePASS).

python tools/convert_datasets_pass/DensePASS_13_op.py  XXXXXX  --nproc  8  #for test

#Open-set SynPASS，WildPASS2K → DensePASS
python tools/convert_datasets_pass/SynPASS_13.py XXXXXX  --nproc  8 --split train --mapping train #for source domain
python tools/convert_datasets_pass/DensePASS_11.py  /home/zyfone/hard-disk/zyf/datasets/DensePASS/DensePASS  --nproc  8 #for test
#Open-set GTA，SynPASS → SynPASS
python tools/convert_datasets_pass/gta_13.py XXXXXX  --nproc  8 #for source domain
python tools/convert_datasets_pass/SynPASS_13.py XXXXXX  --nproc  8 --split val --mapping test #for test
python tools/convert_datasets_pass/SynPASS_13.py XXXXXX  --nproc  8 --split test --mapping test #for test
#Open-set SynPASS，ACDC → ACDC
python tools/convert_datasets_pass/ACDC_13.py XXXXXX  --nproc  8 --split train
python tools/convert_datasets_pass/ACDC_13.py XXXXXX  --nproc  8 --split val
python tools/convert_datasets_pass/ACDC_13.py XXXXXX  --nproc  8 --split test
```


## Setup Environment

```shell
pip install -r requirements.txt 
```
build mmcv form source file
```shell
cd mmcv

pip install -e . -v
```

## Training

For convenience, we provide an   of the final xxxxx.
A training job can be launched using:

```shell
python run_experiments.py --config xxx
```

## Testing & Predictions

```shell
sh test.sh path/to/checkpoint_directory
```




If you have any questions , please contact me at 478756030@qq.com


```BibTeX

@article{zheng2026seeing,
  title={Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation},
  author={Zheng, Yuanfan and Peng, Kunyu and Zheng, Xu and Yang, Kailun},
  journal={arXiv preprint arXiv:2603.15475},
  year={2026}
}

```
