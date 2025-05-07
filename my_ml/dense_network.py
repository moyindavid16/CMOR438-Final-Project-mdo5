import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["DenseNetwork"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def MSE(W, B, X, y):
    total_error = 0
    for xi, yi in zip(X, y):
        _, A = forward_pass(W, B, xi)
        L = len(A) - 1  # get output layer index
        total_error += 0.5 * (A[L] - yi) ** 2
    return total_error / len(X)


def initialize_weights(layers):
    W = dict()
    B = dict()
    for i in range(1, len(layers)):
        W[i] = np.random.randn(layers[i], layers[i-1]) * \
            np.sqrt(2.0 / layers[i-1])
        B[i] = np.zeros((layers[i], 1))
    return W, B


def forward_pass(W, B, x):
    Z = dict()
    A = dict()
    A[0] = x.reshape(-1, 1)
    for i in range(1, len(W) + 1):
        Z[i] = W[i] @ A[i-1] + B[i]
        A[i] = sigmoid(Z[i])
    return Z, A


class DenseNetwork:
    def __init__(self, layers=[4, 8, 1]):
        self.layers = layers
        self.W, self.B = initialize_weights(layers=self.layers)

    def train(self, X_train, y_train, alpha=0.01, epochs=50):
        self.errors_ = [MSE(self.W, self.B, X_train, y_train)]
        print(f"Starting Cost = {self.errors_[0]}")
        L = len(self.layers) - 1
        for k in range(epochs):
            for xi, yi in zip(X_train, y_train):
                Z, A = forward_pass(self.W, self.B, xi)
                deltas = dict()
                deltas[L] = (A[L] - yi) * d_sigmoid(Z[L])
                for i in range(L-1, 0, -1):
                    deltas[i] = (self.W[i+1].T @ deltas[i+1]) * d_sigmoid(Z[i])
                for i in range(1, L+1):
                    self.W[i] -= alpha * deltas[i] @ A[i-1].T
                    self.B[i] -= alpha * deltas[i]
            self.errors_.append(MSE(self.W, self.B, X_train, y_train))
            print(f"{k+1}-Epoch Cost = {self.errors_[-1]}")

    def predict(self, xi):
        _, A = forward_pass(self.W, self.B, xi)
        return A[len(self.layers)-1][0][0]
