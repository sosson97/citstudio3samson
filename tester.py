#tester.py

#Abstract Implementation
import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import env

class Tester():
	"""Class: Tester
		
		Description: 
			Tester class provides functions such as conducting test with trained-model, dumping out the testing result as csv-formatted file.
		
	"""


	#Internal
	def __init__(self, parameters, model):
		"""Function: __init__
			
			Description:
				Initialize testing parameter, trained-model.	

			Args:
				parameters (dic): a dictionary containing testing parameters. 
				model (NN Model): Neural net model such as CNN, LSTM trained by Trainer class.

			Attributes:
				parameters (dic): a dictionary containing testing parameters. 
				model (NN Model): Neural net model such as CNN, LSTM trained by Trainer class.
				result (list): testing result stored line by line. Each line is comma-sepearted for each column.
				loader_data (Torch DataLoader): input data which has a form of Torch DataLoader. It can be directly used as input to test process.

			Returns: 
				None

		"""


		#super.__init__()
		self.parameters = parameters
		self.model = model
		self.result = None
		self.loader_data = self._test_data_load()

	def _test_data_load(self):
		"""Function: _test_data_load

			Description: 
				load test data from given path and make it a form of Torch Tensor(it is a data structure like matrix).  

			Args:
				input_path (str): path to input data

			Returns:
				test_data (DataLoader): test data formatted as torch DataLoader
		"""
		import numpy as np
		from numpy import genfromtxt
		test_data_np = genfromtxt(env.test_input_name, delimiter=',')	
		test_feature_np = np.array([l[env.feature_start_index:env.feature_start_index + env.features_num] for l in test_data_np][1:])
		test_label_np = np.array([l[env.feature_start_index + env.features_num] for l in test_data_np][1:])

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
	def test(self):
		"""Function: test

			Description:
				test input data in self.loader_data using self.model

			Args:
				None

			Returns:
				None but test result is stored in self.result.
		"""

		
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

	def load_model(self, model_path):
		"""Function: load_model
			
			Description: 
				load trained-model which is saved as binary format using pickle.

			Args:
				model_path (str): path to trained-model saved as binary format.
			
			Returns:
				None but self.model now contains trained-model.

		"""
		print("loading trained model")
		f = open(model_path, "rb")
		self.model = pickle.load(f)
		print(model_path + " is loaded")

	#dump is buggy now
	def dump_output(self, dirname_output):
		f = open(dirname_output + "/output.txt", "w")
		for line in self.result:
			f.write(line + '\n')
		f.close()
		print("Dumped test result in " + dirname_output)

