from feature_extractor import FeatureExtractor
import functions
from models import XGBoostModel, SVRModel

import time
import env
import os
import subprocess
from csv import reader


#1. Prepare test data in list.
history_file_name = "raw/1960-2018.csv"
"""1-1. Test file configuraton, creation
Note: Test file must contain the column of Name, WAR, playerid, age, ServiceTime
"""
test_year = 2017
test_file_name "raw/" + test_year + "_test_data.csv"

def filter_by_year(spark, df):
    df = df.filter(df.Season = test_year)
    return df

fe = FeatureExtractor()
fe.raw_to_df(history_file_name)
fe.df_update(filter_by_year)
col = ["Name", "playerid", "WAR", "Age", "ServiceTime"]
fe.df_update(functions.selection, col)
fe.dump_df(test_file_name)

"""1-2. Load test file
"""
test_file = open(test_file_name)
test_csv_reader = reader(test_file, delimiter=',')
next(reader, None)#skip header
test_file.close()

"""Prediction Phase
Baseline Principle: If ServiceTime of one player in test_year is greater or equal than 4, make cluster information using recent 4 ages(X-3 ~ X).
Otherwise, use recent Y ages where Y is ServiceTime of one player in test_year
"""
#Iteration starts 
pred_diff_list = []
for row in test_csv_reader:
    #2. Get cluster information
    test_name = str(row[0])
    test_id = int(row[1])
    test_WAR = float(row[2])
    test_ServiceTime = int(row[3])
    test_age = int(row[4])
    
    """2-1. Set age range from ServiceTime.
    """
    age_high = test_age - 1
    if test_ServiceTime >= 4: 
        age_low = test_age - 4
    else:
        age_low = test_age - test_ServiceTime
       
    """2-2. Create proper WAR enumeration csv.
    Note: data since test_year should be discarded.
    """
    WAR_by_age_file_name = "csv/" + test_year + "_WAR_enumerated_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"
    if not os.path.exists(WAR_by_age_file_name):
        fe = FeatureExtractor()
        fe.raw_to_df(history_file_name)
        def filter_until_year(sparak, df):
            df = df.filter(df.Season < test_year)
            return df
        fe.df_update(filter_until_year)
        fe.df_update(functions.WAR_enumeration_by_age)
        age_range = ["WAR" +  str(i) for i in range(age_low, age_high+1)]
        col = ["Name", "playerid"] + age_range 
        fe.df_update(functions.selection, col)
        fe.dump_df(WAR_by_age_file_name)

    #3. Run Rscript to create csv containing cluster information
    cluster_csv_file_name = "csv/" + test_year + "_test_cluster_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"
    
    if not os.path.exists(cluster_csv_file_name):
        WAR_by_age_file_name_abs = os.getcwd() + WAR_by_age_file_name
        cluster_csv_file_name_abs = os.getcwd() + cluster_csv_file_name
        subprocess.call(["Rscript", "R/kml.R", (age_high - age_low + 1), WAR_by_age_file_name_abs,
                        cluster_csv_file_name_abs], shell=False)
    
    #4. Filter players in same cluster
    """4-1. get clutser of testing player
    Note: From here, all generated csv files are temporary
    """
    fe = FeatureExtractor()
    fe.raw_to_df(cluster_csv_file_name)
    def filter_by_playerid(spark, df):
        df = df.filter(df.playerid = test_id)
        return df
    fe.df_update(filter_by_playerid)
    fe.dump_df("csv/tmp.csv")
    
    test_cluster = 'A'
    with open("csv/tmp.csv") as f:
        csv_reader = reader(f, delimiter=',')
        next(csv_reader, None)#skip header
        row = next(csv_reader) 
        test_cluster = row[-1]

    """4-2 filter by cluster
    """
    fe = FeatureExtractor()
    fe.raw_to_df(histroy_file_name)
    def filter_by_cluster(spark, df):
        df = df.filter(df.Cluster = test_cluster)
        return df
    fe.df_update(filter_by_cluster)
     








###
for player in test_players:
    1. Data preparation for each player
    age = player[age] 
    fe = FeatureExtractor()
    fe.raw_to_df("raw/1960-2018_WAR_enumerated_by_age.csv")
   
    Run update df to create WAR graph of recent 4-ages 
    fe.update_df(WAR_enumerated_by_age)

    Run Rscript to create clusterd csv
    
    2. Cluster Determination
    ??"predict which cluster this player will be in next year" -> 여기가 가정이 적용될 수 있는 부분인가??
    -> you can use double clustering
   

    3. Train / Test xgboost model using player in same cluster


4. Print result based on distribution similarity score
###



