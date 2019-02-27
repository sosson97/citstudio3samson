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
	def dump_output(self, dirname_train, dirname_test):
		train_df, test_df = self.df.randomSplit([0.9, 0.1], seed = 42)
		
		train_df.toPandas().to_csv("train_input/input.csv", header=True)
		train_df.toPandas().to_csv("test_input/input.csv", header=True)
		#train_df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").csv(dirname_train + "/input")
		#test_df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").csv(dirname_test + "/input")
		print("Dumped processed train data in " + dirname_train)
		print("Dumped processed test data in " + dirname_test)

#class: OutputType
class OutputType():
	def __init__(self, player_name, expected_WAR):
		self.player_name = player_name
		self.expected_WAR = expected_WAR

class FeatureExtractorDemo(FeatureExtractor):
	def raw_to_df(self):
		super().raw_to_df()

	def dump_output(self,dirname_train, dirname_test):
		super().dump_output(dirname_train, dirname_test)
		#infile = open(self.filepath_input, "r")
		#outfile = open(dirname_output + "/structured_data.txt", "w")
		#outfile.write(infile.read())



##################
#Custom Functions#
##################
def WAR2014to2016(spark, df):
	df.createOrReplaceTempView('pitcher')
	df = spark.sql('''SELECT Name, playerid, 
										sum(CASE WHEN Season = "2014" THEN WAR ELSE 0 END) 2014WAR,
										sum(CASE WHEN Season = "2015" THEN WAR ELSE 0 END) 2015WAR,
										sum(CASE WHEN Season = "2016" THEN WAR ELSE 0 END) 2016WAR,
										avg(WAR) as last3WAR, max(Age) as Age
										FROM pitcher
										GROUP BY Name, playerid''')
	return df

def join_with_2017(spark, df):
	df_2017 = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("FanGraphs_Leaderboard_2017_Pitcher_Leader.csv")
	df_2017.createOrReplaceTempView('2017pitcher')
	df.createOrReplaceTempView('pitcher')
	df = spark.sql('''SELECT pitcher.2014WAR as 2014WAR, pitcher.2015WAR as 2015WAR, pitcher.2016WAR as 2016WAR,
													 pitcher.Age as Age, 2017pitcher.WAR as 2017WAR
										FROM pitcher, 2017pitcher
										WHERE pitcher.playerid = 2017pitcher.playerid
										''')

	#from pyspark.ml.feature import VectorAssembler
	#all_features = ["2014WAR", "2015WAR", "2016WAR"]
	#assembler = VectorAssembler( inputCols = all_features, outputCol = "features")
	df = df.select("Age", "2014WAR", "2015WAR", "2016WAR", "2017WAR")
	return df


