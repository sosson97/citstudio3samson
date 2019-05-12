import sys
import time	
import os	
import subprocess	
from csv import reader	
from pyspark.sql.functions import col
from time import ctime, strptime

import predictor_path#path configuration
sys.path.append(predictor_path.main_path)
from feature_extractor import FeatureExtractor	
import functions	
from models import XGBoostModel, SVRModel	
import env	

#Util
def file_line_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i+1

def filter_by_membership(spark, df, col_name, l):
    df = df[col(col_name).isin(l)]
    return df

def make_csv_reader_wo_header(file_path):
    f = open(file_path)
    csv_reader = reader(f, delimiter=',')
    header = next(csv_reader, None)
    return header, csv_reader
        
#0. Argument check
if len(sys.argv) < 2:
    print("Argument erorr: Test year is missed. You have to give the target test year as first argument.")
    sys.exit()
if len(sys.argv) < 3:
    print("Argument warning: Any output path is not set. Output will be stored in default path.")

predictor_name = os.path.basename((sys.argv[0])[:-3])
test_year = int(sys.argv[1])
file_prefix = predictor_name + "_" + str(test_year)
print(predictor_name + " starts")
print(str(test_year) + " TEST")


#1. Prepare test / train data
all_players_file_path = predictor_path.raw_file_path
"""1-1. Test file configuraton, creation	
Note: Test file must contain the column of Name, WAR, playerid, age, ServiceTime	
"""	
test_file_name = file_prefix + "_test_data.csv"
test_file_path = os.path.join(predictor_path.test_path, test_file_name)	

train_file_name = file_prefix + "_train_data.csv"
train_file_path = os.path.join(predictor_path.train_path, train_file_name)

#def filter_test_data(spark, df):	
#    df = df.filter(df.Season == test_year)
#    df = df.filter(df.ServiceTime > 1)
#    return df	
def filter_test_data(spark, df):
    new_df = df.filter(df.Season == test_year).select(df.playerid).withColumnRenamed("playerid", "tmpid")
    df = df.filter(df.Season == (test_year-1)).join(new_df, df.playerid == new_df.tmpid, "inner").drop("tmpid")
    return df
def filter_train_data(spark, df):
    df = df.filter(df.Season < test_year-1)
    return df

#test data
print("\033[31m" + "Testing Year: " + str(test_year)  + "\033[0m")
print("\033[31m" + "Test data preparation"  + "\033[0m")
if not os.path.exists(test_file_path):	
    fe = FeatureExtractor()	
    fe.raw_to_df(all_players_file_path)	
    fe.df_update(filter_test_data)
    fe.df_update(functions.null_remover, ["nextWAR"])
    fe.dump_df(test_file_path)	

#train data
print("\033[31m" + "Train data preparation"  + "\033[0m")
if not os.path.exists(train_file_path):	
    fe = FeatureExtractor()	
    fe.raw_to_df(all_players_file_path)	
    fe.df_update(filter_train_data)
    fe.df_update(functions.null_remover, ["nextWAR"])
    fe.dump_df(train_file_path)	

"""1-2. Load test file	
"""
test_header, test_csv_reader = make_csv_reader_wo_header(test_file_path)

"""Prediction Phase	
Baseline Principle: If ServiceTime of one player in test_year is greater or equal than 4, make cluster information using recent 4 ages(X-3 ~ X).	
Otherwise, use recent Y ages where Y is ServiceTime of one player in test_year	
"""	
#output file setting
if not os.path.exists(os.path.join(predictor_path.output_path, predictor_name)):
    os.mkdir(os.path.join(predictor_path.output_path, predictor_name))

if len(sys.argv)==3:
    result_file_path = sys.argv[2]
else:
    parsed_time = strptime(ctime())
    result_file_name = file_prefix + "_" + str(parsed_time.tm_year) + "_" + str(parsed_time.tm_mon) + "_" + str(parsed_time.tm_mday) + "_" + str(parsed_time.tm_hour) + "_" + str(parsed_time.tm_min) + ".csv"
    result_file_path = os.path.join(predictor_path.output_path, predictor_name, result_file_name)
f = open(result_file_path, "w")
f.write('id,real,pred\n')
f.close()

#Iteration starts 
"""5-0-6. Run XGBoost
"""
xgbm = XGBoostModel()
param_map = {
        "feature_start_index":env.feature_start_index,
        "features_num":env.features_num,
        "label_index":env.label_index,
        "id_index":env.id_index,
        "WAR_index":env.WAR_index,
        "metric":"rmse"
        }
from random import randint
xgbm.train(train_file_path, param_map, 1000, randint(0,100))    
xgbm.test(test_file_path, param_map)   
xgbm.dump_output(result_file_path , mode="a", header=False)    

#6. Report Result
"""6-1. Raw Result"""
def join_result(spark, df):
   new_df = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load(test_file_path).select(["Name","playerid"])
   df = df.join(new_df, new_df.playerid == df.id).select(["Name", "playerid", "real", "pred"])
   return df

fe = FeatureExtractor()
fe.raw_to_df(result_file_path)
fe.df_update(join_result)
fe.dump_df(result_file_path)
"""6-2. Distribution Similarity """

"""6-3. Reference: ZiPs """
#End of Iteration
