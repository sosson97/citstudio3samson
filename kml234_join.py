from feature_extractor import FeatureExtractor
import functions
import sys
import os

in2 = sys.argv[1]
in3 = sys.argv[2]
in4 = sys.argv[3]
out = sys.argv[4]

def three_join(spark, df):
    df1 = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load(in2).withColumnRenamed("pred", "kml2").drop("Name")
    df2 = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema","true").load(in3).withColumnRenamed("pred", "kml3").drop("Name")
    df = df.join(df1, df.playerid == df1.playerid).join(df2, df.playerid==df2.playerid).select(["Name", df.playerid, df.real,"kml2", "kml3", "pred"])
    return df

fe = FeatureExtractor()
fe.raw_to_df(in4)
fe.df_update(three_join)
fe.dump_df(out)
