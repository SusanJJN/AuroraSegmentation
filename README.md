# AuroraSegmentation

This is the implementation of A Statistical Analysis of Aurora Evolution during the Substorms using Deep Learning Models. 
This repository includes the codes of two contents, which match the innotation stage and training stage.

# Environment
The code is developed using python 3.8 on Windows 11. NVIDIA GPUs are needed.

# Requirements
* Python 3.8
* CUDA 10.0
* PyTorch 1.8
* mmseg

# Installation
1. Clone this repo:
```python
git clone https://github.com/SusanJJN/AuroraSegmentation.git
```
2. Install dependencies:
For innotate_by_SAM.ipynb, you need to first install dependencies according to the instructions of Segment Anything (https://github.com/facebookresearch/segment-anything).
For HrSeg, you need to first install dependencies according to the instructions of mmsegmentation (https://github.com/open-mmlab/mmsegmentation).

# Data preparation
You can also download all the [original images for training](https://github.com/SusanJJN/AuroraSegmentation/releases/download/v1.0/training_images.rar).

# Innotation
You can follow the instructions in innotate_by_SAM.ipynb. 
The checkpoint used in the notebook can be downloaded in SAM repository.
The [innotated image](https://github.com/SusanJJN/AuroraSegmentation/releases/download/v1.0/innotated_images.rar) can be downloaded as well.

# Training and Inference
1. Split the training set, validation set and test set, and respectively put them in /HrSeg/data/Aurora-dataset.
2. The process of training and inference are shown in /HrSeg/training.ipynb and /HrSeg/inference.ipynb.

# Pretrained model
You can download the [pretrained model](https://github.com/SusanJJN/AuroraSegmentation/releases/download/v1.0/checkpoint_hrseg.pth) here.
