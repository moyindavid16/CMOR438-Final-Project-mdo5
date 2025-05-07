# Supervised Learning Algorithms

This directory contains implementations of various supervised learning algorithms for player analysis.

## Algorithms Implemented

1. **Linear Regression**
   - Used for predicting player market values
   - Features: overall rating, age, league level, contract duration
   - Implementation: Single neuron with identity activation function

2. **Logistic Regression**
   - Used for classifying players as offensive or defensive
   - Features: shooting, defending, pace, dribbling, physical attributes
   - Implementation: Single neuron with sigmoid activation function

3. **Neural Network**
   - Used for predicting player market values
   - Features: overall rating, contract expiry, age, league level
   - Architecture: Multi-layer network with sigmoid activation

4. **AdaBoost**
   - Used for both classification and regression tasks
   - Features: Same as Linear Regression and Logistic Regression
   - Base estimators: Decision trees and linear models

5. **K-Nearest Neighbors (KNN)**
   - Used for player position prediction
   - Features: Player performance attributes
   - Implementation: Distance-based classification

## Dataset Usage

All algorithms use the EA FC 24 dataset with different feature selections based on the task:
- For value prediction: Overall rating, age, league level, contract duration
- For position classification: Performance attributes (pace, shooting, etc.)

## Data Scaling

Data scaling is crucial in this project for several reasons:
1. **Numerical Stability**: Large values can cause overflow in mathematical operations
2. **Equal Feature Importance**: Ensures all features contribute equally to the model
3. **Gradient Descent Optimization**: Helps achieve faster convergence
4. **Algorithm Requirements**: Some algorithms (like neural networks) require normalized inputs

We use StandardScaler from scikit-learn to normalize our features:
- Transforms features to have mean=0 and variance=1
- Applied to both features and target variables in regression tasks
- Essential for algorithms using distance metrics or gradient descent

## Reproducing Results

1. Ensure all dependencies are installed
2. Load the dataset using the data_processing.py utility
3. Run the respective notebook for each algorithm
4. Each notebook contains detailed comments and visualization of results