from feature import FeatureExtractorDemo, OutputType

year = "2014"


#2. cleaning, feature extarction
print("-------------" + "\033[0m")
fed = FeatureExtractorDemo("simple schema", "csv/fangraph_projection_" + year + ".csv")
fed.raw_to_df()

def compare_zips(spark, df):
	df_zips = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("csv/zips_pitchers_" + year +".csv")
	print(df.count())
	print(df_zips.count())
	
	df.createOrReplaceTempView('fang')
	df_zips.createOrReplaceTempView('zips')
	df = spark.sql('''SELECT avg(abs(fang.WAR - zips.WAR)) as diff
										FROM fang, zips
										WHERE fang.playerid = zips.playerid and fang.IP >= 30
										''')
	print(df.count())
	df.show()
	return df

fed.df_update(compare_zips)

print("\033[31m" + "-------------\n" + "\033[0m")


