#models.py

#Abstract Implementation
from abc import ABC, abstractmethod

from torch import nn
#import xgboost as xgb


#class: CNN
#abstract class

#class: CNN
#abstract class
class CNN(ABC):
	#Internal		
	def __init__(self, parameters):
		#super.__init__()
		self.init_model_(parameters)

	
	def init_model_(self, parameters):
		print("CNN model initiated")

	#API
	@abstractmethod
	def forward(self, x):
		print("forwarding CNN model...")

#class: LSTM
#abstract class
class LSTM(ABC):
	#Internal
	def __init__(self, parameters):
		super.__init__()
		init_model_(parameters)

	def init_model_(parameters):
		print("LSTM model initiated")

	#API
	@abstractmethod
	def forward():
		print("forwarding LSTM model...")

#class: XGBoost
#abstract class
class XGBoost(ABC):
	#Internal
	def __init__(self, parameters):
		super.__init__()
		self.model = None
		self.output = []
		init_model_(parameters)
	
	def init_model_(parameters):
		print("XGBoost model initiated")
	
	#API
	@abstractmethod
	def train(train_parameters, dirname_input):
		print("XGBoost training starts")

	
	@abstractmethod
	def test(test_parameters, dirname_input):
		print("XGBoost testing starts")
	
	def dump_output(dirname_output):
		print("Dumped result of testing in " + dirname_output)



class CNNDemo(CNN):
	def forward(self, x):
		avg = 0;
		for val in x:
			avg += val
		return avg/len(x)



