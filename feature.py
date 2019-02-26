#feature.py

#Abstract Implementation
from abc import ABC, abstractmethod

#from pyspark.sql import SparkSession

#class: FeatureExtractor
#abstract class
class FeatureExtractor(ABC):
	#Internal
	def __init__(self, schema):
		super.__init__()
		self.schema = schema
		self.input = None

	#API
	@abstractmethod
	def raw_to_df(dirname_input):
		print("Converting raw data into structured format...")
	
	@abstractmethod
	def dump_output(dirname_output):
		print("Dumped processed data in " + dirname_output)


#class: OutputType
class OutputType():
	def __init__(self, player_name, expected_WAR):
		self.player_name = player_name
		self.expected_WAR = expected_WAR

