# Custom ML Implementations

This directory contains custom implementations of machine learning algorithms used throughout the project.

## Algorithms Implemented

1. **K-Means Clustering** (k_means.py)
   - Custom implementation of the k-means clustering algorithm
   - Features: Euclidean distance calculation, centroid updates
   - Used in unsupervised learning for player position analysis

2. **Dense Neural Network** (dense_network.py)
   - Custom implementation of a multi-layer neural network
   - Features: Sigmoid activation, backpropagation, MSE loss
   - Used in supervised learning for value prediction

## Implementation Details

### K-Means
- Implements standard k-means algorithm with:
  - Random centroid initialization
  - Euclidean distance metric
  - Centroid update mechanism
  - Convergence checking

### Dense Network
- Implements neural network with:
  - Configurable layer architecture
  - Sigmoid activation function
  - Backpropagation with gradient descent
  - Mean Squared Error loss function

## Usage

These implementations are imported and used by the notebooks in the supervised and unsupervised learning directories.

## Reproducing Results

1. Import the required class from the respective module
2. Initialize with appropriate parameters
3. Use fit() and predict() methods as demonstrated in the notebooks