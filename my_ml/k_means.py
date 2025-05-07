import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["KMeans"]

class KMeans:
	def __init__(self, k=3, max_iter=100):
		self.k = k
		self.max_iter = max_iter
		self.centers = None
		
	def distance(self, point, center):
		return np.sqrt(sum((point[j] - center[j])**2 for j in range(len(point))))
	
	def assign_label(self, point):
		distances = np.array([self.distance(point, center) for center in self.centers])
		return np.argmin(distances)
	
	def assign_clusters(self, X):
		return [self.assign_label(x) for x in X]
	
	def update_centers(self, X, labels):
		n_features = X.shape[1]
		new_centers = np.zeros((self.k, n_features))
		counts = np.zeros(self.k)
		
		for i, point in enumerate(X):
			label = labels[i]
			new_centers[label] += point
			counts[label] += 1
			
		# Avoid division by zero
		counts = np.where(counts == 0, 1, counts)
		# print("HEY")
		# print(counts)
		# print(new_centers)
		new_centers = new_centers / counts[:, np.newaxis]
		return new_centers
	
	def fit(self, X):
		# Initialize centers randomly
		indices = np.random.choice(X.shape[0], self.k, replace=False)
		self.centers = X[indices]
		# print(self.centers)
		
		for _ in range(self.max_iter):
			labels = self.assign_clusters(X)
			new_centers = self.update_centers(X, labels)
			# print("NEW", new_centers)
			
			# Check convergence
			if np.allclose(self.centers, new_centers):
				break
				
			self.centers = new_centers
		# print(self.centers)
		
		return self.assign_clusters(X)
	
	def predict(self, X):
		return self.assign_clusters(X)