# Unsupervised Learning Algorithms

This directory contains implementation of clustering algorithms for player analysis.

## Algorithm Implemented

### K-Means Clustering
- Used for discovering natural player position groupings
- Features: pace, shooting, defending, dribbling, physical attributes
- Implementation: Custom k-means algorithm with Euclidean distance
- Number of clusters: 2 (roughly corresponding to offensive/defensive positions)

## Dataset Usage

The algorithm uses the EA FC 24 dataset, focusing on player performance attributes:
- Shooting
- Defending
- Pace
- Dribbling
- Physical attributes

## Data Scaling

Data scaling is essential in K-means clustering for several reasons:
1. **Distance Calculations**: K-means relies on Euclidean distance, which is sensitive to scale
2. **Feature Dominance**: Prevents features with larger ranges from dominating the clustering
3. **Equal Contribution**: Ensures all attributes contribute equally to cluster formation

We use StandardScaler to normalize features:
- Standardizes features to mean=0 and variance=1
- Applied before clustering to ensure fair distance calculations
- Helps discover more meaningful clusters

## Reproducing Results

1. Ensure all dependencies are installed
2. Load the dataset using the data_processing.py utility
3. Run the KMeansClustering.ipynb notebook
4. The notebook includes visualizations of cluster formations and analysis