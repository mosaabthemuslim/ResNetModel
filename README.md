# ResNet Model from Scratch

A PyTorch implementation of the ResNet (Residual Network) architecture, built from scratch for image classification on the CIFAR-10 dataset. This project demonstrates a deep understanding of residual blocks, skip connections, and managing deep neural networks.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [ResNet Architecture](#resnet-architecture)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Repository Contents](#repository-contents)
- [Acknowledgements](#acknowledgements)

## 🚀 Overview

This repository contains a from-scratch implementation of the ResNet architecture, as introduced in the seminal paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He et al.

The project trains a ResNet model on the **CIFAR-10 dataset** to classify images into 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The implementation includes custom data loading, model definition, training loop, and evaluation, all built using PyTorch.

## ✨ Key Features

- **Custom ResNet Implementation:** Built the ResNet model from the ground up using PyTorch, including the fundamental residual blocks.
- **Skip Connections:** Implements identity and convolutional skip connections to solve the vanishing gradient problem in deep networks.
- **CIFAR-10 Dataset:** Utilizes the popular CIFAR-10 dataset for training and evaluation.
- **Training Pipeline:** A complete training loop with loss calculation, backpropagation, and optimizer steps.
- **Evaluation:** Measures model performance by calculating accuracy on a test set.
- **Modular Design:** The code is structured for clarity and easy modification for other datasets or ResNet variants.

## 🏗️ Project Structure

The implementation follows a logical, step-by-step process:

1.  **Environment Setup:** Imports necessary libraries (PyTorch, Torchvision, Matplotlib).
2.  **Data Loading & Preprocessing:** Downloads CIFAR-10 and creates DataLoaders with transformations (normalization, random cropping, flipping).
3.  **Model Architecture:**
    - `BasicBlock`: The fundamental residual block for shallower ResNets (e.g., ResNet18, ResNet34).
    - `ResNet`: The main class that constructs the entire network by stacking layers of BasicBlocks.
4.  **Training Loop:** Defines the loss function (Cross-Entropy), optimizer (SGD with Momentum), and the iterative training process.
5.  **Testing & Evaluation:** Evaluates the trained model on the unseen test set to report final accuracy.

## 🧠 ResNet Architecture

The core idea of ResNet is the use of **residual blocks** with **skip connections**. These connections allow the gradient to flow directly through the network, making it possible to train very deep models effectively.

A basic residual block can be summarized as:
`Output = F(x) + x`

Where:
- `x` is the input to the block.
- `F(x)` represents the learned transformations (convolutional layers, BatchNorm, ReLU).
- The `+ x` is the skip connection that adds the original input to the transformed output.

## 💻 Installation & Usage

### Prerequisites

Ensure you have Python and PyTorch installed. You can install the required packages using pip:

```bash
# It's recommended to use a virtual environment
pip install torch torchvision matplotlib numpy lightning
```

### Running the Project

The entire project is contained within a Jupyter Notebook.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mosaabthemuslim/ResNetModel.git
    cd ResNetModel
    ```

2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook resnetmodel.ipynb
    ```

3.  **Run the cells:** Execute the cells in the notebook sequentially. The notebook will:
    - Automatically download the CIFAR-10 dataset.
    - Define and initialize the ResNet model.
    - Train the model for a set number of epochs.
    - Output the training loss and final test accuracy.

## 📊 Results

After training the model as configured in the notebook, you can expect to achieve a **test accuracy of over 85%** on the CIFAR-10 dataset. This demonstrates the effectiveness of the ResNet architecture even when implemented from scratch.

The training loss will be plotted, showing a steady decrease as the model learns.

*(Note: The final accuracy may vary slightly depending on random initializations and hardware.)*

## 📁 Repository Contents

- `resnetmodel.ipynb`: The main Jupyter Notebook containing the complete code for data loading, model definition, training, and evaluation.
- `README.md`: This file, providing an overview and instructions for the project.

## 🙏 Acknowledgements

- The original ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
- The PyTorch team and community for the excellent deep learning framework.
- CIFAR-10 dataset providers.

---
