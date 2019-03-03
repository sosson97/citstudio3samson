#tester.py

#Abstract Implementation
from abc import ABC, abstractmethod
import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#class: Tester
class Tester():
	#Internal
	def __init__(self, parameters, model):
		#super.__init__()
		self.parameters = parameters
		self.model = model
		self.result = None
		self.loader_data = self.test_data_load_()

	def test_data_load_(self):
		import numpy as np
		from numpy import genfromtxt
		test_data_np = genfromtxt("test_input/input.csv", delimiter=',')	
		test_feature_np = np.array([l[1:5] for l in test_data_np][1:])
		test_label_np = np.array([l[5] for l in test_data_np][1:])

		class testing(Dataset):
			def __init__(self):
				self.len = len(test_feature_np)
				self.x_data = torch.Tensor(test_feature_np)
				self.y_data = torch.Tensor(test_label_np)

			def __getitem__(self, index):
				return self.x_data[index], self.y_data[index]
    
			def __len__(self):
				return self.len
    
		test = testing()
		return DataLoader(test, batch_size=5, shuffle=False)	



	#API
	def test(self, dirname_input):
		print(self.model.__class__.__name__ + " model testing starts...")
		
		from torch.autograd import Variable
		dtype = torch.FloatTensor

		true = []
		pred = []
		for t, (x, y) in enumerate(self.loader_data):
			x_var = Variable(x.type(dtype))
			y_var = Variable(y.type(dtype))
	
			scores = self.model(x_var)
			score_numpy = scores.data.numpy().reshape(-1)
			true_numpy = y.numpy().reshape(-1)
	    
			true = np.concatenate((true, true_numpy))
			pred = np.concatenate((pred, score_numpy))
		
		self.result = np.array(list(zip(true,pred)))
		for tr, pr in self.result:
			print("%.2f" % tr + " " + "%.2f" % pr)
			
		absdiff = 0
		squmean = 0
		for line in self.result:
			if line[0]-line[1] > 0:
				absdiff += line[0]-line[1]
				squmean += (line[0]-line[1])*(line[0]-line[1])
			else:
				absdiff += line[1]-line[0]
				squmean += (line[1]-line[0])*(line[1]-line[0])
		absdiff = absdiff/len(self.result)
		squmean = squmean/len(self.result)
		print("average abs diff = " + str(absdiff))
		print("square mean of diff = " + str(squmean))
		print("testing done!")

	def load_model(self, path_model):
		print("loading trained model")
		f = open(path_model, "rb")
		self.model = pickle.load(f)
		print(path_model + " is loaded")

	#dump is buggy now
	def dump_output(self, dirname_output):
		f = open(dirname_output + "/output.txt", "w")
		for line in self.result:
			f.write(line + '\n')
		f.close()
		print("Dumped test result in " + dirname_output)

"""
class TesterDemo(Tester):
	def test(self, dirname_input):
		super().test(dirname_input)
		f = open(dirname_input + "/toy_test_input.txt", "r")
		test_input = []
		l = f.read().split('\n')[1:-1]
		self.result = []
		for line in l:
			parsed_line = line.split(' ')
			x = [float(parsed_line[1]), float(parsed_line[2]), float(parsed_line[3])]
			y = self.model.forward(x)
			self.result.append(parsed_line[0] + " " + str(y))
		f.close()
		
	def dump_output(self, dirname_output):
		super().dump_output(dirname_output)
	

"""	
