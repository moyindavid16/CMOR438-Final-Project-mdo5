import numpy as np
import pytest
from my_ml.dense_network import DenseNetwork, sigmoid, d_sigmoid, MSE

def test_sigmoid():
    # Test sigmoid function
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.99  # Very large positive number
    assert sigmoid(-100) < 0.01  # Very large negative number
    
    # Test array input
    x = np.array([-1, 0, 1])
    result = sigmoid(x)
    assert np.allclose(result, [0.26894142, 0.5, 0.73105858])

def test_d_sigmoid():
    # Test derivative of sigmoid
    assert d_sigmoid(0) == 0.25  # Maximum value at x=0
    x = np.array([-1, 0, 1])
    result = d_sigmoid(x)
    assert np.allclose(result, [0.196612, 0.25, 0.196612])

def test_network_initialization():
    # Test network initialization with default parameters
    net = DenseNetwork()
    assert len(net.layers) == 3
    assert net.layers == [4, 8, 1]
    
    # Test custom architecture
    custom_layers = [2, 4, 3]
    net = DenseNetwork(layers=custom_layers)
    assert net.layers == custom_layers
    
    # Check weights and biases initialization
    assert len(net.W) == len(custom_layers) - 1
    assert len(net.B) == len(custom_layers) - 1

def test_network_prediction():
    # Test prediction shape and range
    net = DenseNetwork([2, 4, 1])
    X = np.array([1.0, 2.0])
    
    prediction = net.predict(X)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1  # Sigmoid output range

def test_network_training():
    # Test training with simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR function
    
    net = DenseNetwork([2, 4, 1])
    net.train(X, y, alpha=0.1, epochs=10)
    
    # Check if error list is created and decreasing
    assert hasattr(net, 'errors_')
    assert len(net.errors_) == 11  # Initial error + 10 epochs
    assert net.errors_[-1] <= net.errors_[0]  # Error should decrease

def test_mse():
    # Test mean squared error calculation
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    
    # Create a simple network
    net = DenseNetwork([1, 1])
    error = MSE(net.W, net.B, X, y)
    
    assert isinstance(error, float)
    assert error >= 0  # MSE should be non-negative