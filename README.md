# Few-Shot Learning with CLIP and Learnable Contexts

This project implements a few-shot learning approach using OpenAI's CLIP model and learnable contexts to classify images of dried figs into two categories:
- **Super**: Smooth and unbroken surface.
- **Sadyek**: Partially opened, exposing the interior.

The implementation leverages the COOP (Context Optimization) technique, where learnable context tokens are optimized to improve classification performance with minimal labeled data.

## Features
- **Few-Shot Learning**: Train the model with just a few examples per class.
- **CLIP Integration**: Uses the CLIP model for zero-shot and few-shot classification tasks.
- **Learnable Contexts**: A custom learnable context mechanism to adapt textual prompts to specific datasets.
- **Custom Dataset**: Supports a custom dataset of `.bmp` images stored in class-specific directories.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- clip-by-openai
- sklearn
- Pillow

Install dependencies via pip:
```bash

pip install torch torchvision clip-by-openai scikit-learn pillow
```
Place your dataset in the following directory structure:
```

resources/
├── super/
│   ├── image1.bmp
│   ├── image2.bmp
│   └── ...
├── sadyek/
│   ├── image1.bmp
│   ├── image2.bmp
│   └── ...
```
## Usage
1. Clone the repository:
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. Run the script:
```
python Few_Shot_With_COOP.py
```
The script will train the model using the provided images and display the final accuracy.

## Key Components

#### 1. **Custom Dataset**
The `CustomImageDataset` class loads images from the specified directories and applies preprocessing.

#### 2. **Learnable Context Module**
The `LearnableContext` class initializes a set of learnable tokens that are concatenated with the class embeddings during training.

#### 3. **Training Loop**
The script trains the learnable context tokens while keeping the CLIP model frozen.

#### 4. **Evaluation**
The `evaluate` function computes the classification accuracy on the entire dataset.

## Results
After training, the script outputs the classification accuracy on the full dataset. This demonstrates the model's ability to generalize with minimal labeled data.

## Customization
- Adjust the `train_size` variable to modify the number of training examples per class.
- Update the `class_texts` to adapt the textual descriptions for your dataset.
- Modify hyperparameters like `lr`, `num_epochs`, or `context_length` for optimization.

## Acknowledgments
This project is inspired by the CLIP model by OpenAI and the COOP (Context Optimization) methodology.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


