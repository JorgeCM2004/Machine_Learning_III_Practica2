import numpy as np

class F_Preprocessing:
	"""
		Clase que replica StandardScaler de scikit-learn.
	"""
	def __init__(self):
		self.continuous_features = [
			'age', 'creatinine_phosphokinase', 'ejection_fraction',
			'platelets', 'serum_creatinine', 'serum_sodium', 'time'
		]
		self.mean = None
		self.std = None

	def fit(self, X):
		X_vals = X[self.continuous_features].values
		self.mean = np.mean(X_vals, axis=0)
		self.std = np.std(X_vals, axis=0)

	def transform(self, X):
		for i, col in enumerate(self.continuous_features):
			if self.std[i] != 0:
				X[col] = (X[col] - self.mean[i]) / self.std[i]
		return X

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
