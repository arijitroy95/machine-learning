import numpy as np
from itertools import combinations_with_replacement


class DataManipulation:
	def shuffle_data(self, X, y, seed=None):
		"""
			shuffleing data X and y with seed value
		"""
		if seed:
			np.random.seed(seed)
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		return X[idx], y[idx]

	def normalize(self, X, axis=-1, order=2):
		"""
			normalize dataset X
		"""
		l2 = np.atleast_1d(np.linalg,norm(X, order, axis))
		l2[l2 == 0] = 1
		return X / np.expend_dims(l2, axis)

	def standardized(self, X):
		""" standardized dataset X """
		X_std = X
		mean = X.mean(axis=0)
		std = X.std(axis=0)
		for col in range(X.shape[1]):
			if std[col]:
				X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
		return X_std
	
	def polynomial_features(self, X, degree):
		""" returns polynomial features of all the features of given degree """
		n_samples, n_features = X.shape
		def index_combo():
			combs = [combinations_with_replacement(range(n_features), i) for i in range(degree + 1)]
			flat_combs = [item for sublist in combs for item in sublist]
			return flat_combs
		
		combinations = index_combo()
		n_output_features = len(combinations)
		X_new = np.empty((n_samples, n_output_features))
		for i, index_comb in enumerate(combinations):
			X_new[:, i] = np.prod(X[:, index_comb], axis=1)
		return X_new
	
	
	def train_test_split(self, X, y, shuffle=True, seed=None, test_size=0.2):
		""" spliting dataset X, y into two parts training and testing based on test_size"""
		if shuffle:
			X, y = self.shuffle_data(X, y, seed=seed)
		split_idx = len(y) - int(len(y) * test_size)
		X_train, X_test = X[:split_idx], X[split_idx:]
		y_train, y_test = y[:split_idx], y[split_idx:]
		return X_train, X_test, y_train, y_test


class DataOperation:
	def mean_squared_error(self, y, y_pred):
		""" returns mean squared error between y and y predicted values """
		return np.mean(np.power(y - y_pred, 2))
	
	def calculate_variance(self, X):
		""" variance of dataset X """
		mean = np.ones(X.shape) * X.mean(0)
		n_samples = X.shape[0]
		return (1 / n_sampels) * np.diag((X - mean).T.dot(X - mean))
	
	def calculate_std_dev(self, X):
		""" standared deviation of dataset X """
		return np.sqrt(self.calculate_variance(X))
	
	def euclidean_distance(self, x1, x2):
		""" calculate euclidian distance between two vectors """
		return np.linalg.norm(x1 - x2)
	
	def accuracy(self, y, y_pred):
		""" accuracy between y acctual and y predicted """
		return np.sum(y == y_pred, axis=0)  /len(y)

