import numpy as np
import pandas as pd
from src.F_Bayesian_MCMC import Bayesian_MCMC
from src.F_Data_Loader import Data_Loader
from pathlib import Path

if Path("datasets/fallo_cardiaco_train.csv").exists():
	train_df = pd.read_csv("datasets/fallo_cardiaco_train.csv")
else:
	Data_Loader().create_datasets()
	train_df = pd.read_csv("datasets/fallo_cardiaco_train.csv")

X_train = train_df.drop(columns=['DEATH_EVENT']).values
y_train = train_df['DEATH_EVENT'].values

model = Bayesian_MCMC()
ITERATIONS = 50000
STEP_SIZE = 0.1
BURNIN = 0.1

model.fit(X_train, y_train, iterations=ITERATIONS, proposal_width=STEP_SIZE, burnin=BURNIN)

if not Path("model").exists():
	Path("model").mkdir()

np.save('model/model_trace.npy', model.trace)
