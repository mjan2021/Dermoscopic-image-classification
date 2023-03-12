# Dermoscopic Image Classification

This repo is corresponding to the paper (paper-link).



# Data 
We used [HAM10k](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset to train the models. Dataset have the following class distribution
 - mel : 1113
 - bkl : 1099
 - bcc : 514
 - akiec : 327
 - vasc : 142
 - df : 115

## Install
Create a seprate environment and install all dependencies

#### Create Environment
1. Virtual Environment
Create VENV environment
`python -m venv /path/to/environemnt/ `

Activate the environment
`venv\Scripts\activate`


2. Conda
Create Conda Environment
`conda create env --name name-of-envrionment`

Activate Conda Environement
`conda activate name-of-environemnt`


#### Install Dependencies
`pip install -r requirements.txt`


#### Pytorch
Verify that you pytorch is installed and cuda is configured.


```
import torch
torch.cuda.is_available()

```

## Running






## Preprocessing
1. Augmentation
2. Generative Adversrial Networks

## Models to be used

1. Efficientnet
2. ViT
3. ConvNext
4. ResNet50
5. CNN

## Contribution
@malsaidi @mjan2021
## Contact