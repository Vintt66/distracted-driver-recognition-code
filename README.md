# Distracted Driver Recognition

This project contains:

1. `config.py`: dataset, model and training settings.
2. `split_data.py`: creation of the fixed internal train/validation split.
3. `dataset.py`: image loading and preprocessing.
4. `models.py`: Custom CNN, pretrained ResNet18 and pretrained MobileNetV3-Small.
5. `metrics.py`: classification metrics and confusion matrix calculation.
6. `train.py`: training one model with one random seed.
7. `requirements.txt`: Python dependencies.

## Project Description

This project contains development code for ten-class distracted-driver image
classification using the AUC Distracted Driver Dataset V2. The fixed internal
split contains 10,044 training images and 2,511 validation images and is created
by camera and class with seed 42.

