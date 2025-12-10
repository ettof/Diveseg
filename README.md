# [AAAI 2026] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter
[Zhiyang Chen*](https://scholar.google.com.hk/citations?hl=zh-CN&user=02BOfjcAAAAJ), 
[Chen Zhang*](https://scholar.google.com.hk/citations?hl=zh-CN&user=1QgIaYkAAAAJ), 
[Hao Fang](https://scholar.google.cz/citations?user=PA0RVvgAAAAJ) and 
[Runmin Cong](https://scholar.google.cz/citations?user=-VrKJ0EAAAAJ&hl)

[[`paper`](https://arxiv.org/abs/2511.08334)] [[`BibTeX`](#CitingDiveSeg)]

*These authors contributed equally.
<div align="center">
  <img src="DiveSeg.png" width="100%" height="100%"/>
</div><br/>

## Installation

```bash
conda create --name DiveSeg python=3.10 -y
conda activate DiveSeg
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/ettof/Diveseg.git
cd Diveseg
pip install -r requirements.txt
```

## Data Preparation & Pre-trained Weights
Download the two benchmark datasets and organize them as follows:  
```
data/
├── UIIS/
│   ├── train/
│   ├── val/
│   └── annotations/
│       ├── train.json
│       └── val.json
└── USIS10K/
    ├── multi_class_annotations/
    ├── foreground_annotations/
    ├── train/
    ├── val/
    └── test/
```
- **[UIIS Dataset](https://github.com/LiamLian0727/WaterMask)**
- **[USIS10K Dataset](https://github.com/LiamLian0727/USIS10K)**

The pre-trained weights of DINOv2 are available for download at [link](https://github.com/facebookresearch/dinov2). (We use the register-free version of DINOv2-Large.)

Please place the downloaded files in the [checkpoints](checkpoints) directory as specified in the config file.

##  Train & Evaluate
Train the DiveSeg model on UIIS or USIS10K dataset:
```bash
bash train.sh
```
Evaluate pre-trained models on test sets: 
```bash
bash eval.sh
```

You are expected to get results like this:

| Dataset |      Test      | Backbone | $mAP$ | $AP_{50}$ | $AP_{75}$ |                                           weights                                           |
|:-------:|:--------------:|:--------:|:-----:|:---------:|:---------:|:-------------------------------------------------------------------------------------------:| 
|  UIIS   |    Instance    |  ViT-L   | 35.6  |   52.0    |   38.5    | [model](https://drive.google.com/file/d/1hrho8Jg6SNypazx_8xg486rRfyUOaCtK/view?usp=drive_link) |
| USIS10K | Class-Agnostic |  ViT-L   | 64.1  |   82.8    |   72.2    | [model](https://drive.google.com/file/d/1xSAqrooEUKwJ4xRxBcFvXGI894QO_DIH/view?usp=drive_link) |
| USIS10K |  Multi-Class   |  ViT-L   | 48.4  |   62.3    |   54.4    | [model](https://drive.google.com/file/d/1mt8pgCIq1s6HtQKIs22yBfn8jGCBNM47/view?usp=drive_link) |

## <a name="CitingDiveSeg"></a>Citing DiveSeg
```BibTeX
@article{chen2025empowering,
  title={Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter},
  author={Chen, Zhiyang and Zhang, Chen and Fang, Hao and Cong, Runmin},
  journal={arXiv preprint arXiv:2511.08334},
  year={2025}
}
```

## Acknowledgement

This repo is based on [DINOv2](https://github.com/facebookresearch/dinov2), [detectron2](https://github.com/facebookresearch/detectron2) and
[Mask2Former](https://github.com/facebookresearch/Mask2Former). Thanks for their great work!
