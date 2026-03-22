<div align="center">
  <h1><b>Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation</b></h1>

  <a href="https://github.com/zyfone/EDA-PSeg"><img src="https://img.shields.io/github/stars/zyfone/EDA-PSeg?color=yellow" alt="stars"></a>
  <a href="https://github.com/zyfone/EDA-PSeg/network/members"><img src="https://img.shields.io/github/forks/zyfone/EDA-PSeg?color=lightblue" alt="forks"></a>
  <a href="https://github.com/zyfone/EDA-PSeg/commits/main"><img src="https://img.shields.io/github/last-commit/zyfone/EDA-PSeg?color=green" alt="last-commit"></a>
  <img src="https://img.shields.io/badge/PRs-Welcome-blue" alt="PRs Welcome">

  <br><br>

  <p>
    <a href="https://arxiv.org/abs/"><strong>Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation</strong></a>
  </p>

  <p>
    <a href="https://github.com/zyfone/">Yuanfan Zheng<sup>1</sup></a>, 
    <a href="https://scholar.google.com/citations?user=pA9c0YsAAAAJ&hl=en">Kunyu Peng<sup>2</sup></a>, 
    <a href="https://zhengxujosh.github.io/">Xu Zheng<sup>3</sup></a>, 
    <a href="https://yangkailun.com/">Kailun Yang*<sup>1</sup></a>
  </p>

  <p>
    <sup>1</sup>Hunan University; 
    <sup>2</sup>IAR, Karlsruher Institut für Technologie; 
    <sup>3</sup>Hong Kong University of Science and Technology (HKUST)
  </p>
</div>


## Data Preparation

download link:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [DensePASS](https://github.com/chma1024/DensePASS)
- [SynPASS](https://drive.google.com/file/d/1u-5J13CD6MXpWB53apB-L6kZ3hK1JR77/view?usp=sharing)
- [DensePASS (WildPASS2K + DensePASS)](https://github.com/chma1024/DensePASS)
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


## :pencil:Related repos
Our project references the codes in the following repos:

* [DAFormer](https://github.com/lhoyer/DAFormer)

other code :
* https://github.com/KU-VGI/BUS
* https://github.com/InSAI-Lab/Trans4PASS
* https://github.com/RUCAIBox/EulerFormer
* https://github.com/CityU-AIM-Group/SIGMA




If you have any questions , please contact me at 478756030@qq.com


```BibTeX

@article{zheng2026seeing,
  title={Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation},
  author={Zheng, Yuanfan and Peng, Kunyu and Zheng, Xu and Yang, Kailun},
  journal={arXiv preprint arXiv:2603.15475},
  year={2026}
}

```
