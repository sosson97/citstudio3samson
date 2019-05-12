from feature_extractor import FeatureExtractor
import functions

"""
def three_join(spark, df):
    df1 = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load("output/kml2_strp/kml2_strp_2017_2019_5_6_19_11.csv").withColumnRenamed("pred", "kml2").drop("Name")
    df2 = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema","true").load("output/kml3_strp/kml3_strp_2017_2019_5_6_18_30.csv").withColumnRenamed("pred", "kml3").drop("Name")
    df = df.join(df1, df.playerid == df1.playerid).join(df2, df.playerid==df2.playerid).select(["Name", df.playerid, df.real,"kml2", "kml3", "pred"])
    return df

fe = FeatureExtractor()
fe.raw_to_df("output/kml4_strp/kml4_strp_2017_2019_5_6_17_49.csv")
fe.df_update(three_join)
fe.dump_df("output/kml_strp234_aggregated.csv")
"""
"""
def zips_join(spark, df, year):
    name = "raw/zips_"+str(year) +".csv"
    new_df = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema","true").load(name)
    df = df.join(new_df, new_df.Player == df.Name, "inner").select([df.Name, df.playerid, "zWAR", "real"])
    return df

fe = FeatureExtractor()
for i in [2017]:
    fe.raw_to_df("output/allip3_kml234_strp_joined_" + str(i) +".csv")
    fe.df_update(zips_join, i)
    fe.dump_df("zips_joined_" + str(i) + ".csv")
"""

def last1_join(spark, df, year):
    name = "output/last1ml_exp/last1ml_null_" +str(year) +".csv"
    new_df = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema","true").load(name)
    df = df.select(["playerid"]).withColumnRenamed("playerid","id")
    df = df.join(new_df, new_df.playerid == df.id, "inner")
    return df

fe = FeatureExtractor()
for i in range(2013,2019):
    fe.raw_to_df("output/allip3_kml234_strp_joined_" + str(i) +".csv")
    fe.df_update(last1_join, i)
    fe.dump_df("output/last1ml_null_joined_" + str(i) + ".csv")


"""
def join_with_lastyear(spark, df, year):
    df1 = df.filter(df.Season==year).withColumnRenamed("WAR", "real").withColumnRenamed("playerid","playerid1").select(["Name", "playerid1", "real"])
    df2 = df.filter(df.Season==(year-1)).withColumnRenamed("WAR", "pred").withColumnRenamed("playerid","playerid2").select(["Name", "playerid2", "pred"])
    new_df = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load("output/allip3_kml234_strp_joined_" + str(year) + ".csv").withColumnRenamed("playerid", "playerid3").select("playerid3")

    df = df1.join(df2, df1.playerid1==df2.playerid2).join(new_df, df1.playerid1==new_df.playerid3)
    return df

fe = FeatureExtractor()
for i in [2018,2014,2013]:
    fe.raw_to_df("raw/1960-2018_allip3.csv")
    fe.df_update(join_with_lastyear,i)
    fe.dump_df("output/last_year_prediction_"+str(i)+".csv")
"""

"""
fe = FeatureExtractor()
fe.raw_to_df("raw/2017_test_base_data.csv")
fe.df_update(functions.join, "output/predictor_using_kml/third_bug_fixed.csv", "playerid", "left_outer")
fe.dump_df("output/2017_joined_third_test.csv")
"""

#fe.raw_to_df("raw/from_25_to_28_clusters_K.csv")
#fe.df_update(functions.join_age_WAR, 29)
#fe.dump_df("raw/from_25_to_28_clusters_K_test.csv")



"""
col = ["Name", "playerid"]
fe.raw_to_df("raw/1960-2018_WAR_enumerated_by_age.csv")
for i in range(25,29):
    fe.df_update(functions.null_remover, ["WAR"+str(i)])
    col.append("WAR"+str(i))
fe.df_update(functions.selection,col)
fe.dump_df("raw/players_from_25_to_28.csv")
"""

"""
for i in range(1,16):
    fe.raw_to_df("raw/1960-2018_WAR_enumerated.csv")
    fe.df_update(functions.null_remover, ["WAR"+str(i)])
    col.append("WAR"+str(i))
    fe.df_update(functions.selection,col)
    fe.dump_df("raw/players_until_career_" + str(i) + ".csv")
"""

#fe.raw_to_df("raw/1960-2018.csv")
#fe.df_update(functions.WAR_enumeration_by_age)
#fe.dump_df("raw/1960-2018_WAR_enumerated_by_age.csv")
#fe.df_update(null_remover)
#fe.dump_df("nullrm.csv")

#path_list = ["train_input/train_simple_1year.csv", "test_input/test_simple_1year.csv"]
#fe.dump_df(None, True, test_2017_train_less2017_split, path_list)


#Trial 1
"""
fe = FeatureExtractor()
fe.raw_to_df("raw/last1year.csv")

fe.df_update(clustering, 3)
path_list = []
for i in range(3):
    path_list.append("raw/clustered_last1year" + str(i) + ".csv")
fe.dump_df(None, True, cluster_split, path_list, 3)
"""
