#feature.py

#Abstract Implementation
from abc import ABC, abstractmethod

from pyspark.sql import SparkSession
import time

#class: FeatureExtractor
#abstract class
class FeatureExtractor(ABC):
	#Internal
	def __init__(self, schema, filepath_input):
		#super.__init__()
		self.schema = schema
		self.filepath_input = filepath_input
		self.spark = SparkSession.builder.master('local[4]').appName('SparkSQL_Review').getOrCreate()
		self.df = None
	#API
	@abstractmethod
	def raw_to_df(self):
		print("Converting raw data into structured format...")
		self.df = self.spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(self.filepath_input)
	
	def df_update(self, custom_function):
		self.df = custom_function(self.spark, self.df)

	@abstractmethod
	def dump_output(self, dirname_output):
		self.df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").csv(dirname_output + "/input")
		print("Dumped processed data in " + dirname_output)


#class: OutputType
class OutputType():
	def __init__(self, player_name, expected_WAR):
		self.player_name = player_name
		self.expected_WAR = expected_WAR

class FeatureExtractorDemo(FeatureExtractor):
	def raw_to_df(self):
		super().raw_to_df()

	def dump_output(self,dirname_output):
		super().dump_output(dirname_output)
		#infile = open(self.filepath_input, "r")
		#outfile = open(dirname_output + "/structured_data.txt", "w")
		#outfile.write(infile.read())



##################
#Custom Functions#
##################
def WAR2014to2016(spark, df):
	df.createOrReplaceTempView('pitcher')
	df = spark.sql('''SELECT Name, playerid, avg(WAR) as last3WAR, Age
										FROM pitcher
										GROUP BY Name, playerid''')
	return df
