#tester.py

#Abstract Implementation
from abc import ABC, abstractmethod

#from torch import nn

#class: Tester
#abstract class
class Tester(ABC):
	#Internal
	def __init__(self, parameters, model):
		super.__init__()
		self.parameters = parameters
		self.model = model

	#API
	@abstractmethod
	def test(dirname_input):
		print(model.__class__.__name__ + " model testing starts...")
	
	@abstractmethod
	def dump_output(dirname_output):
		print("Dumped test result in " + dirname_output)
