# Few-Shot Learning with CLIP for Image Classification
This project demonstrates the use of few-shot learning with OpenAI's CLIP model to improve recall for a binary image classification task. Images are stored in two directories, each representing a separate class, and we aim to optimize the model's recall on these specific categories.

## Project Overview
In this notebook, we utilize CLIP's capabilities to perform few-shot learning, which is particularly useful when only a limited number of examples are available for each class. By leveraging CLIP’s pre-trained visual and textual embeddings, we can efficiently differentiate between two classes of images with minimal labeled data.

## Objectives
Apply few-shot learning using CLIP to classify images into two distinct classes.
Maximize recall for each class through model tuning.
Provide performance evaluation metrics for the classification.
Dataset Structure
The dataset is expected to be organized as follows:

resources/
├── super/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── sadyek/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

  super/: Contains images belonging to Class 1.
  sadyek/: Contains images belonging to Class 2.

## Requirements
To run this notebook, ensure you have the following dependencies installed:

. Python 3.x
. Jupyter Notebook
. PyTorch
. OpenAI's CLIP model
. NumPy
. Matplotlib (for plotting and visualization)

You can install the required libraries with the following command:
pip install torch numpy matplotlib openai-clip

## Notebook Structure
Setup and Imports: Load necessary libraries and set up CLIP for image embedding.
Data Loading: Load and preprocess images from each class directory.
Feature Extraction: Use CLIP to extract embeddings for each image.
Few-Shot Learning: Apply a few-shot learning technique by selecting a small number of images from each class as exemplars.
Classification and Tuning:
Use embeddings to classify images into the two classes.
Optimize for maximum recall, adjusting parameters as necessary.
Evaluation: Measure the model's performance using metrics such as recall, precision, and F1 score.

## results
The notebook provides performance metrics, which include recall for each class. You can modify parameters in the notebook to achieve better recall results for the two classes.
