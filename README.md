# MNIST Handwritten Digit Classification with PyTorch

## Overview

This project implements three different neural network architectures to classify handwritten digits from the famous MNIST dataset. The models range from a simple linear classifier to a more sophisticated convolutional neural network (CNN), demonstrating the progression in accuracy that comes with more advanced architectures.

## Table of Contents
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a 28x28 grayscale pixel array.

Key characteristics:
- Input shape: 1x28x28 (channels × height × width)
- 10 output classes (digits 0-9)
- Balanced class distribution

## Models

### 1. MNISTModelV0 - Basic Linear Classifier
- **Architecture**:
  - Flatten layer (784 input features)
  - Linear layer (784 → 10)
  - Linear layer (10 → 10)
- **Characteristics**:
  - No activation functions
  - Simple feedforward structure
  - Fast training but limited accuracy

### 2. MNISTModelV1 - Neural Network with ReLU Activations
- **Architecture**:
  - Flatten layer (784 input features)
  - Linear layer (784 → 10) + ReLU
  - Linear layer (10 → 10) + ReLU
- **Characteristics**:
  - Introduces non-linearity with ReLU
  - Better learning capacity than V0
  - Still relatively simple

### 3. MNISTModelV2 - Convolutional Neural Network (CNN)
- **Architecture**:
  - **Convolutional Block 1**:
    - Conv2d (1→10 channels, 3x3 kernel)
    - ReLU
    - Conv2d (10→10 channels, 3x3 kernel)
    - ReLU
    - MaxPool2d (2x2)
  - **Convolutional Block 2**:
    - Conv2d (10→10 channels, 3x3 kernel)
    - ReLU
    - Conv2d (10→10 channels, 3x3 kernel)
    - ReLU
    - MaxPool2d (2x2)
  - **Classifier**:
    - Flatten
    - Linear (490 → 10)
- **Characteristics**:
  - Leverages spatial information through convolutions
  - Multiple layers of feature extraction
  - Pooling for dimensionality reduction
  - Highest accuracy of the three models

## Results

| Model          | Test Accuracy | Test Loss | Training Time (3 epochs) |
|----------------|---------------|-----------|--------------------------|
| MNISTModelV0   | 91.94%        | 0.2867    | ~1 minute                |
| MNISTModelV1   | 73.27%        | 0.9604    | ~1 minute                |
| MNISTModelV2   | 98.57%        | 0.0461    | ~5 minutes               |

**Key Observations**:
- The CNN (ModelV2) significantly outperforms the other models
- ModelV1 performed worse than ModelV0, suggesting the architecture might need tuning
- All models achieve reasonable accuracy, demonstrating the relative simplicity of MNIST

## Usage

### Training the Models

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook MNIST_Image_classification.ipynb
```

### Making Predictions

The notebook includes helper functions to:
- Train and evaluate models
- Visualize predictions
- Compare model performance

Example prediction code:
```python
# Make predictions on test samples
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=model_2, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- pandas
- tqdm
- torchmetrics
- mlxtend (version 0.19.0 or higher)

Install all requirements with:
```bash
pip install torch torchvision matplotlib numpy pandas tqdm torchmetrics mlxtend
```

## Visualizations

The notebook includes several visualizations:

1. **Sample Images**: Displays random samples from the dataset
2. **Training Progress**: Shows loss and accuracy during training
3. **Model Comparison**: Bar chart comparing test accuracies
4. **Prediction Examples**: Visualizes model predictions with true labels

Example visualization code:
```python
# Plot predictions
plt.figure(figsize=(6, 6))
for i, sample in enumerate(test_samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = pred_classes[i]
    truth_label = test_labels[i]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    color = "g" if pred_label == truth_label else "r"
    plt.title(title_text, fontsize=10, c=color)
    plt.axis(False)
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Potential improvements:
- Add more advanced CNN architectures
- Implement data augmentation
- Add hyperparameter tuning
- Include confusion matrices
- Add support for other datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Yann LeCun et al. for the MNIST dataset
- The open source community for various utility functions
