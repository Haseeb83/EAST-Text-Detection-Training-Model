# EAST Text Detection Training Pipeline

This repository contains a complete training pipeline for the EAST (Efficient and Accurate Scene Text Detection) model, implemented in PyTorch and designed to run in Google Colab. The project includes a simulated dataset, a simplified EAST model, a full training loop, and visualization of key metrics (loss difference, accuracy, precision, and recall).

## Project Overview

The EAST model is a deep learning-based approach for detecting text in natural scene images. This project provides a starting point for training EAST, with placeholder components (simulated data and a basic model) that you can replace with your own dataset and EAST implementation. The pipeline computes training and validation metrics over multiple epochs and visualizes them in a 2x2 plot grid.

### Features
- **Training Loop**: Full training and validation loop with loss and metric computation.
- **Metrics**: Tracks loss, accuracy, precision, and recall for both training and validation phases.
- **Visualization**: Plots loss difference, accuracy, precision, and recall across epochs.
- **Modularity**: Easily adaptable to custom EAST models and datasets.

## Prerequisites

- **Google Colab**: The code is designed to run in a Colab environment with GPU support.
- **Python Packages**: Required libraries are installed via the notebook:
  - `torch`, `torchvision`
  - `opencv-python`, `matplotlib`, `seaborn`
  - `albumentations`, `transformers`
  - `googletrans==3.1.0a0`, `scikit-learn`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/east-text-detection.git
   cd east-text-detection
