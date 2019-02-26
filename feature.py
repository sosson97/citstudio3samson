#feature.py

#Abstract Implementation
from abc import ABC, abstractmethod

#from pyspark.sql import SparkSession

#class: FeatureExtractor
#abstract class
class FeatureExtractor(ABC):
	#Internal
	def __init__(self, schema, filepath_input):
		#super.__init__()
		self.schema = schema
		self.filepath_input = filepath_input

	#API
	@abstractmethod
	def raw_to_df(self, dirname_input):
		print("Converting raw data into structured format...")
	
	@abstractmethod
	def dump_output(self, dirname_output):
		print("Dumped processed data in " + dirname_output)


#class: OutputType
class OutputType():
	def __init__(self, player_name, expected_WAR):
		self.player_name = player_name
		self.expected_WAR = expected_WAR

class FeatureExtractorDemo(FeatureExtractor):
	def raw_to_df(self, dirname_input):
		pass

	def dump_output(self,dirname_output):
		infile = open(self.filepath_input, "r")
		outfile = open(dirname_output + "/structured_data.txt", "w")
		outfile.write(infile.read())

