# METR 

This is the official implementation of the paper "[Language-aware Multiple Datasets Detection Pretraining for DETRs]([https://arxiv.org/abs/2304.03580](https://www.sciencedirect.com/science/article/abs/pii/S0893608024004301))". 

Authors: Jing Hao*, Song Chen*

## Installation
We use the environment same to DINO to run METR. If you have run DN-DETR or DAB-DETR or DINO, you can skip this step.
We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well.

1. Clone this repo
```sh
git clone https://github.com/isbrycee/METR.git
cd METR
```

2. Install Pytorch and torchvision

Follow the instruction on https://pytorch.org/get-started/locally/.
```sh
# an example:
conda install -c pytorch pytorch torchvision
```

3. Install other needed packages
```sh
pip install -r requirements.txt
```

4. Compiling CUDA operators
```sh
cd models/metr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## Data
Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

## Run
We use METR 4-scale model trained for 12 epochs as default experiment setting.

### Training

```sh
bash scripts_METR/METR_train_dist_4scale_r50_coco.sh

```

### Evaluation
```
bash scripts_METR/METR_eval_dist_4scale_r50_coco.sh
```
Notes:
1. You should change the dataset path on scripts before running.
2. This code implementation also supports for ViT backbone.

# Links
Our model is based on [DINO](https://arxiv.org/abs/2203.03605).


# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```bibtex
@article{hao2024language,
  title={Language-aware multiple datasets detection pretraining for detrs},
  author={Hao, Jing and Chen, Song},
  journal={Neural Networks},
  pages={106506},
  year={2024},
  publisher={Elsevier}
}
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
