import pandas as pd
import numpy as np
from src.F_Preprocessing import F_Preprocessing

class Data_Loader:
	def __init__(self):
		self.preprocessor = F_Preprocessing()

	def create_datasets(self, output_filename_base='fallo_cardiaco', split_ratio=0.8):
		"""
		Divide y normaliza los conjuntos de datos.
		Crea dos nuevos archivos CSV con los conjuntos de datos.
		"""
		df = pd.read_csv("datasets/fallo_cardiaco.csv")
		y = df[["DEATH_EVENT"]]
		X_df_raw = df.drop(columns=["DEATH_EVENT"])

		indices = np.random.permutation(len(y))
		split_idx = int(len(y) * split_ratio)

		train_idx, test_idx = indices[:split_idx], indices[split_idx:]

		X_train_raw, X_test_raw = X_df_raw.iloc[train_idx].copy(), X_df_raw.iloc[test_idx].copy()
		y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

		X_train_norm = self.preprocessor.fit_transform(X_train_raw)
		X_test_norm = self.preprocessor.transform(X_test_raw)

		train_df_processed = pd.concat([X_train_norm, y_train], axis=1)
		test_df_processed = pd.concat([X_test_norm, y_test], axis=1)

		train_filename = f"datasets/{output_filename_base}_train.csv"
		test_filename = f"datasets/{output_filename_base}_test.csv"

		train_df_processed.to_csv(train_filename, index=False)
		test_df_processed.to_csv(test_filename, index=False)

		return train_filename, test_filename

if __name__ == "__main__":
	Data_Loader().create_datasets()
