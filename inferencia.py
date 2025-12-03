import numpy as np
import pandas as pd
from src.F_Bayesian_MCMC import Bayesian_MCMC

trace = np.load('model/model_trace.npy')

test_df = pd.read_csv("datasets/fallo_cardiaco_test.csv")
X_test = test_df.drop(columns=['DEATH_EVENT']).values
y_test = test_df['DEATH_EVENT'].values

model = Bayesian_MCMC(trace=trace)

probs = model.predict(X_test)
preds = (probs > 0.5).astype(int)

print(f"Accuracy: {np.mean(preds == y_test):.2%}")
