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
if len(sys.argv) != 2:
    print("Argument erorr: Test year is missed. You have to give the target test year as first argument.")
    sys.exit()
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
    df = df.filter(df.Season < test_year)
    return df

#test data
print("\033[31m" + "Testing Year: " + str(test_year)  + "\033[0m")
print("\033[31m" + "Test data preparation"  + "\033[0m")
if not os.path.exists(test_file_path):	
    fe = FeatureExtractor()	
    fe.raw_to_df(all_players_file_path)	
    fe.df_update(filter_test_data)
    fe.dump_df(test_file_path)	

#train data
print("\033[31m" + "Train data preparation"  + "\033[0m")
if not os.path.exists(train_file_path):	
    fe = FeatureExtractor()	
    fe.raw_to_df(all_players_file_path)	
    fe.df_update(filter_train_data)	
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
    
parsed_time = strptime(ctime())
result_file_name = file_prefix + "_" + str(parsed_time.tm_year) + "_" + str(parsed_time.tm_mon) + "_" + str(parsed_time.tm_mday) + "_" + str(parsed_time.tm_hour) + "_" + str(parsed_time.tm_min) + ".csv"
result_file_path = os.path.join(predictor_path.output_path, predictor_name, result_file_name)
f = open(result_file_path, "w")
f.write('id,real,pred\n')
f.close()

#Iteration starts 
for row in test_csv_reader:	
    #2. Get cluster information	
    test_name = str(row[test_header.index("Name")])	
    test_id = int(row[test_header.index("playerid")])	
    test_WAR = float(row[test_header.index("WAR")])	
    test_ServiceTime = int(row[test_header.index("ServiceTime")])	
    test_age = int(row[test_header.index("Age")])

    print("\033[31m" +"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Player Name: " + test_name)
    print("Age: " + str(test_age))
    print("Active Years: " + str(test_ServiceTime) + "\033[0m")
    print("")
    """2-1. Set age range from ServiceTime.	
    Note: Here is the place that set the range of data to cluster players!
    """
    range_years = 3
    age_high = test_age	
    if test_ServiceTime >= range_years: 	
        age_low = test_age - (range_years-1)
    else:	
        age_low = test_age - (test_ServiceTime-1)

    """2-2. Create proper WAR enumeration csv.	
    Note: data since test_year should be discarded.	
    """
    print("\033[31m" + "Age " + str(age_low) + "~" + str(age_high) + " data will be used to make prediction" + "\033[0m")
    WAR_by_age_file_name = file_prefix + "_WAR_enumerated_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"	
    WAR_by_age_file_path = os.path.join(predictor_path.internal_path, WAR_by_age_file_name)
    if not os.path.exists(WAR_by_age_file_path):	
        fe = FeatureExtractor()	
        fe.raw_to_df(train_file_path)	
        fe.df_update(functions.WAR_enumeration_by_age)	
        age_range = ["WAR" +  str(i) for i in range(age_low, age_high+1)]
        cols = ["Name", "playerid"] + age_range 	
        fe.df_update(functions.selection, cols)
        fe.df_update(functions.null_remover)
        fe.dump_df(WAR_by_age_file_path)	

    #3. Run Rscript to create csv containing cluster information	
    cluster_csv_file_name = file_prefix + "_test_cluster_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"	
    cluster_csv_file_path = os.path.join(predictor_path.internal_path, cluster_csv_file_name)

    if not os.path.exists(cluster_csv_file_path):	
        lines = file_line_len(WAR_by_age_file_path)
        if lines > 100:
            cluster_num = 15
        elif lines > 7:
            cluster_num = int(lines / 7)
        else:
            print("Sample is too small. Player Name: " + test_name)
            continue
        subprocess.call(["Rscript", os.path.join(predictor_path.main_path, "R/kml.R"), str(age_high - age_low + 1), WAR_by_age_file_path,	
                        cluster_csv_file_path, str(cluster_num)], shell=False)	
    #4. Filter players in same cluster	
    """4-1. get clutser of testing player	
    Note: From here, all generated csv files are temporary	
    """	
    fe = FeatureExtractor()	
    fe.raw_to_df(cluster_csv_file_path)	
    fe.df_update(filter_by_membership, "playerid", test_id)
    
    tmp_file_path = os.path.join(predictor_path.internal_path, "tmp.csv")
    fe.dump_df(tmp_file_path)	

    test_cluster = 'A' 
    _, tmp_reader = make_csv_reader_wo_header(tmp_file_path)
    row = next(tmp_reader, None) 
    if row==None:#missing data
            print("Missing data. Player Name: " + test_name)
            continue
    test_cluster = row[-1]	
   

    print("\033[31m" + test_name + " is in Cluster " + test_cluster + "\033[0m")

    """4-2 filter by cluster	
    """	
    fe = FeatureExtractor()	
    fe.raw_to_df(cluster_csv_file_path)	
    fe.df_update(filter_by_membership, "Cluster", test_cluster)	
    
    tmp_same_cluster_file_path = os.path.join(predictor_path.internal_path, "tmp_same_cluster.csv")
    fe.dump_df(tmp_same_cluster_file_path)

    #5. Make prediction using multiple hypothesis!
    """5-2-1. Double Clustering
    """
    tmp_double_cluster_file_name = "tmp_double_cluster.csv"
    tmp_double_cluster_file_path = os.path.join(predictor_path.internal_path, tmp_double_cluster_file_name)
     
    lines = file_line_len(tmp_same_cluster_file_path)
    if lines > 35:
        cluster_num = 7
    elif lines > 5:
        cluster_num = int(lines / 5)
    else:
        cluster_num  = 1
    subprocess.call(["Rscript", os.path.join(predictor_path.main_path, "R/kml.R"), str(age_high - age_low + 1), tmp_same_cluster_file_path,	
                    tmp_double_cluster_file_path, str(cluster_num)], shell=False)	

    """5-2-2. get clutser of testing player	
    """	

    fe = FeatureExtractor()	
    fe.raw_to_df(tmp_double_cluster_file_path)	
    fe.df_update(filter_by_membership, "playerid", test_id)	
    
    tmp_file_path = os.path.join(predictor_path.internal_path, "tmp.csv")
    fe.dump_df(tmp_file_path)	
    
    test_cluster = 'A' 
    _, tmp_reader = make_csv_reader_wo_header(tmp_file_path)
    row = next(tmp_reader, None) 
    
    if row==None:#missing data
        print("Missing data. Player Name: " + test_name)
        continue
    test_cluster = row[-1]	

    """filter by cluster	
    """	
    fe = FeatureExtractor()	
    fe.raw_to_df(tmp_double_cluster_file_path)	
    fe.df_update(filter_by_membership, "Cluster", test_cluster)

    tmp_same_double_cluster_file_name = "tmp_same_double_cluster.csv"
    tmp_same_double_cluster_file_path = os.path.join(predictor_path.internal_path, tmp_same_double_cluster_file_name)
    fe.dump_df(tmp_same_double_cluster_file_path)


    """5-2-3. Preparing train data with major_cluster
    """
    fe = FeatureExtractor()
    fe.raw_to_df(tmp_same_double_cluster_file_path)
    fe.df_update(functions.selection, "playerid")
    
    tmp_major_ids_file_name = "tmp_major_ids.csv"
    tmp_major_ids_file_path = os.path.join(predictor_path.internal_path, tmp_major_ids_file_name)
    fe.dump_df(tmp_major_ids_file_path)
    
    #Filtering test data by playerid(excerpted from cluser information) and age
    _, tmp_major_ids_reader = make_csv_reader_wo_header(tmp_major_ids_file_path)
    id_list = []
    for row in tmp_major_ids_reader:
        print(type(row[0]))
        id_list.append(int(float(row[0])))
    
    ###IMPORTANT###
    id_list.remove(test_id)
    ###############

    fe = FeatureExtractor()
    fe.raw_to_df(train_file_path)
    fe.df_update(filter_by_membership, "playerid", id_list)
    fe.df_update(filter_by_membership, "Age", age_high)
    #IP ratio filtering!
    def filter_by_IPratio(spark, df):
        df = df.filter(df.ratioIP > 0.7)
        df = df.filter(df.ratioIP < 1.3)
        return df
    fe.df_update(filter_by_IPratio)

    tmp_train_file_name = "tmp_train.csv"
    tmp_train_file_path = os.path.join(predictor_path.internal_path, tmp_train_file_name) 
    fe.dump_df(tmp_train_file_path)
    if file_line_len(tmp_train_file_path)==1:
        print("Lack of good training data. Passed")
        continue
    
    """5-0-5. Preparing test data
    """
    fe = FeatureExtractor()
    fe.raw_to_df(test_file_path)
    fe.df_update(filter_by_membership, "playerid", test_id)
    
    tmp_test_file_name = "tmp_test.csv"
    tmp_test_file_path = os.path.join(predictor_path.internal_path, tmp_test_file_name) 
    fe.dump_df(tmp_test_file_path)

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
    xgbm.train(tmp_train_file_path, param_map, 1000, randint(0,100))    
    xgbm.test(tmp_test_file_path, param_map)   
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
