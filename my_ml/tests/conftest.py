import pytest
import numpy as np

@pytest.fixture
def sample_data_2d():
    """Fixture providing simple 2D dataset with clear clusters"""
    return np.array([
        [0, 0], [0.1, 0], [0, 0.1],  # Cluster 1
        [2, 2], [2.1, 2], [2, 2.1]   # Cluster 2
    ])

@pytest.fixture
def sample_regression_data():
    """Fixture providing simple regression dataset"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])  # Linear relationship
    return X, y