# Satellite Image Classification

This project implements a satellite image classification model using deep learning techniques. The model is built with PyTorch and employs a simple Convolutional Neural Network (CNN) architecture to classify satellite images into predefined categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to classify satellite images using a CNN. The dataset consists of labeled satellite images, which are processed and fed into the model for training. After training, the model can classify new images based on the learned patterns.

## Features

* **Image Preprocessing:** Images are resized and normalized for consistent input to the model.
* **Convolutional Neural Network (CNN):** A simple yet effective architecture is used for image classification.
* **Training and Evaluation:** The model is trained and evaluated using a training/test split of the dataset.
* **GPU Support:** The model leverages CUDA for GPU acceleration if available.

## System Requirements

+ **Python:** 3.6 or higher
+ **PyTorch:** 1.7 or higher
+ **Torchvision:** 0.8 or higher
+ **Google Colab:** Optional for running the notebook online
+ **CUDA:** Required for GPU support (optional)


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guruttamsv/Satellite-Image-Classification.git
cd Satellite-Image-Classification
```
2. Set up a virtual environment (Using conda or virtualenv is recommended):
```bash
# If using Conda
conda create -n satellite-classification python=3.8
conda activate satellite-classification
```
3. Install required packages:
```bash
pip install torch torchvision
```

## Usage

1. **Upload the dataset:** Ensure you have a dataset of satellite images in the required format. The dataset should be structured such that each class of images is in its respective folder within a parent directory named data.
2. **Run the notebook:** Open the ImageClassification.ipynb notebook in Jupyter or Google Colab and execute the cells sequentially.

## Model Architecture

The CNN model is defined as follows:
```python

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Additional layers can be added as needed
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(401408, 128),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

## Training and Evaluation
The training and evaluation of the model are conducted in the notebook. The training loop includes loss calculation and optimization using the Adam optimizer. Evaluation metrics, such as accuracy, are printed at the end of each epoch.

### Example of Training Loop:
```python

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # Accumulate loss for the epoch

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Print average training loss for the epoch
    train_loss /= len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}')


```

## Results

The model's performance is evaluated on the test dataset, and accuracy metrics are printed to gauge the classification performance.

```python
print(f'Accuracy on the dataset: {100 * correct / total:.2f}%')
```

## Limitations and Future Work

* **Limited Dataset:** The model's performance is highly dependent on the size and quality of the dataset.
* **Model Complexity:** More complex models or architectures can be explored for potentially better accuracy.
* **Hyperparameter Tuning:** Additional tuning of hyperparameters can enhance model performance.

## Acknowledgements

* **PyTorch:** For providing the framework for building and training deep learning models.
* **Torchvision:** For the image processing utilities.
* **Google Colab:** For providing an accessible platform to run Jupyter notebooks with GPU support.

