#trainer.py

#Abstract Implementation
from abc import ABC, abstractmethod
import pickle

#from torch import nn

#class: Trainer
#abstract class
class Trainer(ABC):
	#Internal
	def __init__(self, parameters, optimizer, loss_function, model):
		#super.__init__()
		self.parameters = parameters
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.model = model

	#API
	@abstractmethod
	def train(self, dirname_input):
		print(model.__class__.__name__ + " model testing starts...")
	
	def dump_model(self, dirname_output, model_name):
		print("Dumped trained model in " + dirname_output)
		f = open(dirname_output + "/" + model_name, "wb")
		pickle.dump(self.model, f)

class TrainerDemo(Trainer):
	def train(self, dirname_input):
		pass
	
