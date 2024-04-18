# Dynamic Knowledge Adapter with Probabilistic Calibration for Generalized Few-Shot Semantic Segmentation

## Description
This repository contains separate environments for training and testing machine learning models, each with its own requirements file for ease of setup and independent use.

## Setup

### Environment Setup
To ensure each part of the project runs smoothly, the training and testing environments are separated. Install the required dependencies for each by navigating to the respective directory and running:

'''pip install -r requirements.txt'''

### Dataset Preparation
1. **Divide the dataset** into trainset and valset.

### Data Processing
2. **Run data processing**:
   Execute data_processing.py to crop each 1024x1024 image and its corresponding label into four 512x512 images.
   '''python data_processing.py'''

## Training
3. **Start training**:
   Navigate to the training directory and run:
   '''bash train.sh'''

## Testing
4. **Conduct testing**:
   A pre-trained .pth file has been provided for immediate testing. It is stored in the Test/pretrain/ directory.
   '''bash test.sh'''
