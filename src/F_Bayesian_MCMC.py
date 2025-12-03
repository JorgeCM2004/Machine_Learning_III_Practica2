import numpy as np
from scipy.stats import norm

class Bayesian_MCMC:
	def __init__(self, trace = None):
		self.trace = trace

	def _sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def _log_likelihood(self, w, X, y):
		z = np.dot(X, w)
		return np.sum(y * z - np.log(1 + np.exp(z)))

	def _log_prior(self, w, prior_mu=0, prior_sigma=10):
		return np.sum(norm.logpdf(w, prior_mu, prior_sigma))

	def fit(self, X, y, iterations = 5000, proposal_width = 0.1, burnin = 0):
		if 0 < burnin < 1:
			burnin = int(burnin * iterations)
		else:
			burnin = int(burnin)
		# MCMC notebook alfredo
		w = np.zeros(X.shape[1])
		samples = [w]
		for i in range(iterations):
			w_new = w + np.random.normal(scale=proposal_width, size=w.shape)
			log_acceptance_ratio = (self._log_likelihood(w_new, X, y) + self._log_prior(w_new)) - (self._log_likelihood(w, X, y) + self._log_prior(w))
			if np.log(np.random.rand()) < log_acceptance_ratio:
				w = w_new
			samples.append(w)
		self.trace = np.array(samples[burnin:])

	def predict(self, X):
		if not self.trace.any():
			raise ValueError("Entrena el modelo antes.")
		mean_w = np.mean(self.trace, axis=0)
		logits = np.dot(X, mean_w)
		return self._sigmoid(logits)
