from feature_extractor import FeatureExtractor
result_file_path = "/home/sam95/CD3/simple/output/allip3_exp/allip3_kml4_strp_2017.csv"
test_file_path = "/home/sam95/CD3/simple/internal/test/kml4_strp_2017_test_data.csv"
def join_result(spark, df):
   new_df = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load(test_file_path).select(["Name","playerid"])
   df = df.join(new_df, new_df.playerid == df.id).select(["Name", "playerid", "real", "pred"])
   return df

fe = FeatureExtractor()
fe.raw_to_df(result_file_path)
fe.df_update(join_result)
fe.dump_df(result_file_path)
