# Dataset EDA 

Dataset
Dataset used in this study was downloaded form Kaggle and is available at this link [HAM10k](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 

The dataset is highly imbalanced with class "NV" consisting of 67% of the samples. This leads to biased classification towards the majority class by the model, which is a major problem. 

This figure showing the original data distribution highlights the imbalance in the dataset. 

![Original Distribution](../images/OriginalDistribution.png "Original Data Distribution")


## Data Augmentation

Data was augmented was done using torchvision transforms. Script for data augmentation can be found in [GAN-Augmentation](../notebooks/GAN-Augmentation.ipynb)

![Data Augmentation](../images/Augmentation.png "Data Augmentation")

![Augmentation Distribution](../images/AugmentedDistribution.png "Original Data Distribution")

## Synthetic Data

Synthetic Data was generated using AC-GAN

![Synthetic Data](../images/epoch_50_generated_batch.jpg "Synthetic Data")

![GAN Distribution](../images/SyntheticDistribution.png "Original Data Distribution")