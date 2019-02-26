#tester.py

#Abstract Implementation
from abc import ABC, abstractmethod
import pickle
import time
#from torch import nn

#class: Tester
#abstract class
class Tester(ABC):
	#Internal
	def __init__(self, parameters, model):
		#super.__init__()
		self.parameters = parameters
		self.model = model
		self.result = None
	#API
	@abstractmethod
	def test(self, dirname_input):
		print(self.model.__class__.__name__ + " model testing starts...")
		time.sleep(0.5)
		print(". . .")
		time.sleep(0.5)
		print(". . .")
		time.sleep(0.5)
		print("testing done!")

	def load_model(self, path_model):
		print("loading trained model")
		time.sleep(0.5)
		print(". . .")
		time.sleep(0.5)
		print(". . .")
		time.sleep(0.5)
		print(". . .")
		print("loaded!")
		f = open(path_model, "rb")
		self.model = pickle.load(f)
		print(path_model + " is loaded")

	@abstractmethod
	def dump_output(self, dirname_output):
		print("Dumped test result in " + dirname_output)

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
		f = open(dirname_output + "/output.txt", "w")
		for line in self.result:
			f.write(line + '\n')
		f.close()
	


	
