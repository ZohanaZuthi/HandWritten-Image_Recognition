# Handwritten Image Classification

This project focuses on classifying handwritten digits from the popular MNIST dataset using machine learning techniques.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model](#model)
6. [Results](#results)
7. [License](#license)

## Overview

The goal of this project is to build a machine learning model to classify handwritten digits from the MNIST dataset. We use deep learning techniques, particularly Convolutional Neural Networks (CNNs), to achieve high accuracy in recognizing digits from images.

## Dataset

This project uses the **MNIST** dataset, which contains 28x28 grayscale images of handwritten digits (0-9). The dataset is divided into a training set of 60,000 images and a test set of 10,000 images.

- Training Data: 60,000 images labeled with their corresponding digit.
- Test Data: 10,000 images labeled with their corresponding digit.

You can download the dataset from [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

## Installation

Follow these steps to set up the environment and run the project.

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/handwritten-image-classification.git
   cd handwritten-image-classification
   ```

2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This will install libraries like TensorFlow, Keras, NumPy, Matplotlib, etc.

## Usage

To train the model:

```bash
python train_model.py
```

This will train a CNN model on the MNIST dataset and save the model weights to a file `model.h5`.

To evaluate the model:

```bash
python evaluate_model.py
```

This will load the trained model and evaluate its performance on the test dataset, printing out the accuracy score.

To make predictions:

```bash
python predict.py --image path_to_image
```

Replace `path_to_image` with the path to a handwritten image you want to classify.

## Model

The model uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: 28x28 grayscale images.
- **Conv2D Layer**: 32 filters, kernel size 3x3, activation function ReLU.
- **MaxPooling Layer**: Pool size 2x2.
- **Conv2D Layer**: 64 filters, kernel size 3x3, activation function ReLU.
- **MaxPooling Layer**: Pool size 2x2.
- **Flatten Layer**: Converts the 2D matrix into a 1D vector.
- **Dense Layer**: 128 units, activation function ReLU.
- **Output Layer**: 10 units (for the 10 digit classes), activation function softmax.

## Results

Once the model is trained, you should be able to achieve an accuracy of around **98%** on the MNIST test set. 

Example output after running `evaluate_model.py`:

```
Test Loss: 0.04
Test Accuracy: 98.6%
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

