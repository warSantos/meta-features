import json

import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import (stratfied_cv, save_mfs)


class DatasetCent(Dataset):

	def __init__(self, data):
		super().__init__()

		x = data[0].toarray()
		self.x = torch.from_numpy(x)
		self.c = data[1]

		self.n_samples = self.x.shape[0]

		# Building "labels" to the encoder.
		self.centroids = {}
		labels = set(self.c)
		for label in labels:
			self.centroids[label] = np.mean(x[self.c == label], axis=0)

		y = []
		for c in self.c:
			y.append(self.centroids[c])

		self.y = torch.from_numpy(np.array(y))

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples


class EncoderModel(nn.Module):

	def __init__(self, input_size, hidden_layers=3):
		
		super().__init__()
		self.input_size = input_size
		self.output_size = input_size

		self.encoder = nn.ModuleList()
		"""
		for i in range(hidden_layers):
			self.encoder.append(nn.Linear(self.input_size, self.output_size))
		self.encoder.append(nn.Tanh())
		"""

		self.encoder = nn.Sequential(
			nn.Linear(self.input_size, self.output_size),
			nn.Linear(self.input_size, self.output_size),
			nn.Linear(self.input_size, self.output_size),
			nn.Tanh()
		)

	def forward(self, x):
		return self.encoder(x)


class Encoder:

	def __init__(self, input_size, hidden_layers=3):

		self.input_size = input_size
		self.device = torch.device(
			'cuda' if torch.cuda.is_available() else 'cpu')
		self.model = EncoderModel(
			self.input_size, hidden_layers=hidden_layers).to(self.device)

	def fit(self, data: DatasetCent, epochs=30, batch_size=16):

		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		train_loader = DataLoader(
			dataset=data, batch_size=batch_size, shuffle=True)

		# Train the model
		n_total_steps = len(train_loader)
		for epoch in range(epochs):
			for i, (x, y) in enumerate(train_loader):

				x = x.to(self.device)
				y = y.to(self.device)

				# Forward pass
				outputs = self.model(x.float())
				loss = criterion(outputs.float(), y.float())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i+1) % 100 == 0:
					print(
						f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}', end="\r")

	def predict(self, X: torch.Tensor):

		with torch.no_grad():
			X_pred = []
			for x in X.to(self.device):
				pred = self.model(x.float())
				X_pred.append(pred)

		return np.array(torch.stack(X_pred).cpu().numpy())


class EncoderMFs:

	def transform(self, X_train, y_train, X_test, params):

		data_loader = DatasetCent((X_train, y_train))
		enc = Encoder(X_train.shape[1], params["hidden_layers"])
		enc.fit(data_loader)
		return enc.predict(torch.from_numpy(X_test.todense()))

	def build_features(self, dset, params):

		dir_rep = params["dir_rep"].replace("__dset__", dset)
		# For the first cross-val level.
		for fold in np.arange(10):

			fold_recip = dir_rep.replace("__fold__", str(fold))
			train_file = fold_recip.replace("__train_test__", "train")
			reps_train = load_svmlight_file(train_file)
			X_train = reps_train[0]
			y_train = reps_train[1]
			# List of train mfs.
			train_mfs = []
			# Make new splits to generate train MFs.
			splits = stratfied_cv(
				X_train, y_train, dset=dset, fold=fold, load_splits=False)
			for inner_fold in splits.itertuples():

				inner_X_train = X_train[inner_fold.train]
				inner_y_train = y_train[inner_fold.train]
				inner_X_test = X_train[inner_fold.test]

				new_mfs = self.transform(
					inner_X_train, inner_y_train, inner_X_test, params)
				train_mfs.append(new_mfs)

			train_mfs = np.vstack(train_mfs)

			# Generating test meta-features.
			test_file = fold_recip.replace("__train_test__", "test")
			reps_test = load_svmlight_file(test_file)
			X_test = reps_test[0]
			test_mfs = self.transform(X_train, y_train, X_test, params)

			params_prefix = f"""rep/{params["rep"]}/hidden_layers/{params["hidden_layers"]}"""

			save_mfs(dset, "encoder", fold, train_mfs,
					 test_mfs, params_prefix=params_prefix)

	#def build(self, datasets=["webkb"], option="1"):
	def build(self, datasets=["20ng", "reut", "acm"], option="1"):

		with open("data/configs/encoder/params.json", "r") as fd:
			params = json.load(fd)[option]
			for dset in datasets:
				self.build_features(dset, params)


"""
from src.encoder_mfs.encoder import EncoderMFs
enc = EncoderMFs()
enc.build()
"""
