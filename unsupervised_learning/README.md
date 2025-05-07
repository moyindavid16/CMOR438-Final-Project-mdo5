# Unsupervised Learning Algorithms

This directory contains implementation of clustering and dimensionality reduction algorithms for player analysis.

## Algorithms Implemented

### K-Means Clustering
- Used for discovering natural player position groupings
- Features: pace, shooting, defending, dribbling, physical attributes
- Implementation: Custom k-means algorithm with Euclidean distance
- Number of clusters: 2 (roughly corresponding to offensive/defensive positions)

### Principal Component Analysis (PCA)
- Used for dimensionality reduction and feature analysis of player attributes
- Features: pace, shooting, defending, dribbling, physical attributes
- Implementation: Custom PCA using Singular Value Decomposition (SVD)
- Key components:
  - Data centering
  - SVD computation for finding principal components
  - Explained variance ratio calculation
  - Dimension reduction transformation

## Dataset Usage

The algorithms use the EA FC 24 dataset, focusing on player performance attributes:
- Shooting
- Defending
- Pace
- Dribbling
- Physical attributes

## Data Scaling

Data scaling is essential in both algorithms for several reasons:
1. **Distance Calculations**: K-means relies on Euclidean distance, which is sensitive to scale
2. **Feature Dominance**: Prevents features with larger ranges from dominating the analysis
3. **Equal Contribution**: Ensures all attributes contribute equally to the analysis
4. **PCA Requirements**: PCA assumes features are on similar scales for meaningful variance analysis

We use StandardScaler to normalize features:
- Standardizes features to mean=0 and variance=1
- Applied before clustering to ensure fair distance calculations
- Essential for PCA to properly capture variance in the data

## Reproducing Results

1. Ensure all dependencies are installed
2. Load the dataset using the data_processing.py utility
3. Run the respective notebooks (KMeansClustering.ipynb or PCA.ipynb)
4. The notebooks include visualizations and analysis of results