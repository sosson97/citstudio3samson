#trainer.py

#Abstract Implementation
from abc import ABC, abstractmethod

#from torch import nn

#class: Trainer
#abstract class
class Trainer(ABC):
	#Internal
	def __init__(self, parameters, optimizer, loss_function):
		#super.__init__()
		self.parameters = parameters
		self.optimizer = optimizer
		self.loss_function = loss_fuction
		self.model = None

	#API
	@abstractmethod
	def train(self, dirname_input):
		print(model.__class__.__name__ + " model testing starts...")
	
	@abstractmethod
	def dump_output(self, dirname_output):
		print("Dumped trained model in " + dirname_output)

class TrainerDemo(Trainer):
	def train(self, dirname_input):
		pass
	
	def dump_output(self, dirname_output):
		pass
