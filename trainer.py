#trainer.py

#Abstract Implementation
from abc import ABC, abstractmethod
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#class: Trainer
class Trainer():
	#Internal
	def __init__(self, parameters, optimizer, loss_function, model):
		#super.__init__()
		self.parameters = parameters
		self.model = model
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.1)
		self.loss_function = nn.L1Loss()
		self.loader_data = self.train_data_load_()

	def train_data_load_(self):
		import numpy as np
		from numpy import genfromtxt
		train_data_np = genfromtxt("train_input/input.csv", delimiter=',')	
		train_feature_np = np.array([l[1:5] for l in train_data_np][1:])
		train_label_np = np.array([l[5] for l in train_data_np][1:])

		class training(Dataset):
			def __init__(self):
				self.len = len(train_feature_np)
				self.x_data = torch.Tensor(train_feature_np)
				self.y_data = torch.Tensor(train_label_np)

			def __getitem__(self, index):
				return self.x_data[index], self.y_data[index]
    
			def __len__(self):
				return self.len
    
		train = training()
		return DataLoader(train, batch_size=5, shuffle=False)	

	#API
	def train(self, dirname_input, num_epochs):
		print(self.model.__class__.__name__ + " model training starts...")
		
		dtype = torch.FloatTensor

		for epoch in range(num_epochs):
			cost = 0.0
			for t, (x, y) in enumerate(self.loader_data):
				x_var = x.type(dtype)
				y_var = y.type(dtype)
				scores = self.model(x_var)
				
				print(scores)
				print(y_var)
				loss = self.loss_function(scores, y_var)
				cost += loss.data[0]
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			costs.append(cost / (t + 1))
			if (epoch + 1) % (num_epochs / 10) == 0:
				print('Epoch = %d, loss = %.4f' % (epoch + 1, costs[-1]))
		 
		print("training done!")

	def dump_model(self, dirname_output, model_name):
		print("Dumped trained model in " + dirname_output)
		f = open(dirname_output + "/" + model_name, "wb")
		pickle.dump(self.model, f)

"""
class TrainerDemo(Trainer):
	def train(self, dirname_input):
		super().train(dirname_input)
"""
	
