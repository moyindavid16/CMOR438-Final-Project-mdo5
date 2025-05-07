# Data Processing

This directory contains utilities for loading and processing the EA FC 24 dataset.

## Dataset Description

The EA FC 24 Complete Player Dataset contains comprehensive information about football players, including:
- Player attributes (pace, shooting, passing, etc.)
- Physical characteristics
- Club and contract information
- Market values
- Position information

## Data Loading

The data_processing.py module provides utilities for loading the dataset:
- Uses kagglehub for dataset access
- Returns a pandas DataFrame
- Handles data type conversions automatically

## Data Preprocessing

Various preprocessing steps are applied in different analyses:
1. Feature selection based on the specific task
2. Handling missing values
3. Data scaling and normalization
4. Target variable creation for specific tasks

## Data Scaling

Data scaling is a crucial preprocessing step applied throughout the project:
1. **Why We Scale**:
   - Ensures all features contribute equally to the analysis
   - Prevents numerical instability in calculations
   - Required for many ML algorithms to work effectively
   - Improves convergence in gradient-based methods

2. **Scaling Methods Used**:
   - StandardScaler: Standardizes features to mean=0, variance=1
   - Applied to both features and targets in regression tasks
   - Essential for distance-based algorithms and neural networks

## Usage

1. Import the load_fifa_data function
2. Call the function to get the DataFrame
3. Apply necessary preprocessing steps as shown in the notebooks