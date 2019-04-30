from feature_extractor import FeatureExtractor	
import functions	
from models import XGBoostModel, SVRModel	
import env	

import time	
import os	
import subprocess	
from csv import reader	
from pyspark.sql.functions import col

#Util
def file_line_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i+1

def filter_by_membership(spark, df, col_name, l):
    df = df[col(col_name).isin(l)]
    return df



#1. Prepare test / train data
history_file_name = "raw/1960-2018_edited.csv"	
"""1-1. Test file configuraton, creation	
Note: Test file must contain the column of Name, WAR, playerid, age, ServiceTime	
"""	
test_year = 2017	
test_file_name = "raw/" + str(test_year) + "_test_data.csv"	
test_base_file_name = "raw/" + str(test_year) + "_test_base_data.csv"#store last year's WAR
train_file_name = "raw/" + str(test_year) + "_train_data.csv"

def filter_test_data(spark, df):	
    df = df.filter(df.Season == test_year)
    df = df.filter(df.ServiceTime > 1)
    return df	
def filter_test_base_data(spark, df):
    df = df.filter(df.Season == (test_year-1))
    return df
def filter_train_data(spark, df):
    df = df.filter(df.Season < test_year)
    return df

#test data
print("\033[31m" + "Testing Year: " + str(test_year)  + "\033[0m")
print("\033[31m" + "Test data preparation"  + "\033[0m")
if not os.path.exists(test_file_name):	
    fe = FeatureExtractor()	
    fe.raw_to_df(history_file_name)	
    fe.df_update(filter_test_data)	
    cols = ["Name", "playerid", "WAR", "Age", "ServiceTime"]	
    fe.df_update(functions.selection, cols)	
    fe.dump_df(test_file_name)	

    #test base data
    fe = FeatureExtractor()	
    fe.raw_to_df(history_file_name)	
    fe.df_update(filter_test_base_data)	
    #cols = ["Name", "playerid", "WAR"]	
    #fe.df_update(functions.selection, cols)	
    fe.dump_df(test_base_file_name)	


#train data
print("\033[31m" + "Train data preparation"  + "\033[0m")
if not os.path.exists(train_file_name):	
    fe = FeatureExtractor()	
    fe.raw_to_df(history_file_name)	
    fe.df_update(filter_train_data)	
    fe.dump_df(train_file_name)	

"""1-2. Load test file	
"""	
test_file = open(test_base_file_name)	
test_csv_reader = reader(test_file, delimiter=',')	
next(test_csv_reader, None)#skip header	

"""Prediction Phase	
Baseline Principle: If ServiceTime of one player in test_year is greater or equal than 4, make cluster information using recent 4 ages(X-3 ~ X).	
Otherwise, use recent Y ages where Y is ServiceTime of one player in test_year	
"""	
#Iteration starts 	
pred_diff_list = []	
for row in test_csv_reader:	
    #2. Get cluster information	
    test_name = str(row[1])	
    test_id = int(row[env.id_index])	
    test_WAR = float(row[env.WAR_index])	
    test_ServiceTime = int(row[37])	
    test_age = int(row[3])

    print("\033[31m" +"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Player Name: " + test_name)
    print("Age: " + str(test_age))
    print("Active Years: " + str(test_ServiceTime) + "\033[0m")
    print("")
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
    print("\033[31m" + "Age " + str(age_low) + "~" + str(age_high) + " data will be used to make prediction" + "\033[0m")
    WAR_by_age_file_name = "csv/" + str(test_year) + "_WAR_enumerated_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"	
    if not os.path.exists(WAR_by_age_file_name):	
        fe = FeatureExtractor()	
        fe.raw_to_df(train_file_name)	
        fe.df_update(functions.WAR_enumeration_by_age)	
        age_range = ["WAR" +  str(i) for i in range(age_low, age_high+1)]
        print(age_range)
        cols = ["Name", "playerid"] + age_range 	
        fe.df_update(functions.selection, cols)
        fe.df_update(functions.null_remover)
        fe.dump_df(WAR_by_age_file_name)	

    #3. Run Rscript to create csv containing cluster information	
    cluster_csv_file_name = "csv/" + str(test_year) + "_test_cluster_by_age_from_" + str(age_low) + "_to_" + str(age_high) + ".csv"	

    if not os.path.exists(cluster_csv_file_name):	
        WAR_by_age_file_name_abs = os.getcwd() + "/" + WAR_by_age_file_name	
        cluster_csv_file_name_abs = os.getcwd() + "/"  + cluster_csv_file_name	
        lines = file_line_len(WAR_by_age_file_name_abs)
        if lines > 100:
            cluster_num = 15
        elif lines > 20:
            cluster_num = int(lines / 7)
        else:
            print("Sample is too small. Player Name: " + test_name)
            continue
        subprocess.call(["Rscript", "R/kml.R", str(age_high - age_low + 1), WAR_by_age_file_name_abs,	
                        cluster_csv_file_name_abs, str(cluster_num)], shell=False)	
    #4. Filter players in same cluster	
    """4-1. get clutser of testing player	
    Note: From here, all generated csv files are temporary	
    """	

    fe = FeatureExtractor()	
    fe.raw_to_df(cluster_csv_file_name)	
    fe.df_update(filter_by_membership, "playerid", test_id)	
    fe.dump_df("csv/tmp.csv")	

    test_cluster = 'A'	
    with open("csv/tmp.csv") as f:	
        csv_reader = reader(f, delimiter=',')	
        next(csv_reader, None)#skip header	
        row = next(csv_reader, None) 
        if row==None:#missing data
            print("Missing data. Player Name: " + test_name)
            continue
        test_cluster = row[-1]	
   

    print("\033[31m" + test_name + " is in Cluster " + test_cluster + "\033[0m")

    """4-2 filter by cluster	
    """	
    fe = FeatureExtractor()	
    fe.raw_to_df(cluster_csv_file_name)	
    fe.df_update(filter_by_membership, "Cluster", test_cluster)	
    fe.dump_df("csv/tmp_same_cluster.csv")

    #5. Make prediction using multiple hypothesis!
    hp_num = 0
    if hp_num == 0:
        """Scenario 0. Base hypothesis: Cluster prediction w/ majority voitng  
                                        Use last 1 year data
        """
        """5-0-1. Make cluster information w/ 1 year shifting 
        """
        WAR_by_age_file_name_tmp = "csv/tmp_shift_WAR_by_age.csv"	
        fe = FeatureExtractor()	
        fe.raw_to_df(train_file_name)	
        fe.df_update(functions.WAR_enumeration_by_age)	
        age_range = ["WAR" +  str(i) for i in range(age_low+1, age_high+2)]#1 year shifted age!
        print(age_range)
        cols = ["Name", "playerid"] + age_range 	
        fe.df_update(functions.selection, cols)
        fe.df_update(functions.null_remover)
        fe.dump_df(WAR_by_age_file_name_tmp)	

        cluster_csv_file_name_tmp = "csv/tmp_cluster.csv"	

        WAR_by_age_file_name_tmp_abs = os.getcwd() + "/" + WAR_by_age_file_name_tmp	
        cluster_csv_file_name_tmp_abs = os.getcwd() + "/"  + cluster_csv_file_name_tmp	
        lines = file_line_len(WAR_by_age_file_name_tmp_abs)
        if lines > 100:
            cluster_num = 15
        else:
            cluster_num = int(lines / 7)
        subprocess.call(["Rscript", "R/kml.R", str(age_high - age_low + 1), WAR_by_age_file_name_tmp_abs,	
                    cluster_csv_file_name_tmp_abs, str(cluster_num)], shell=False)
        
        """5-0-2. Join shifted cluster data with same cluster file. 
        """
        fe = FeatureExtractor()
        fe.raw_to_df("csv/tmp_same_cluster.csv")
        def inner_join(spark, df):
            new_df =  spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema",
                            "true").load(cluster_csv_file_name_tmp)
            new_df = new_df.withColumnRenamed('Cluster', 'NewCluster').withColumnRenamed('playerid', 'NewPlayerid')
            df = df.join(new_df, df.playerid == new_df.NewPlayerid, "inner")
            return df
        fe.df_update(inner_join)
        fe.dump_df("csv/tmp_joined.csv")
        fe.df_update(functions.selection,["playerid", "NewCluster"])
        fe.dump_df("csv/tmp_cluster.csv")

        """5-0-3. Find majority cluster.
        """
        def find_majority(l):
            myMap = {}
            maximum = ( '', 0 ) # (occurring element, occurrences)
            for n in l:
                if n in myMap: myMap[n] += 1
                else: myMap[n] = 1
                # Keep track of maximum on the go
                if myMap[n] > maximum[1]: maximum = (n,myMap[n])
            return maximum

       	with open("csv/tmp_cluster.csv") as f:
            csv_reader = reader(f)
            next(csv_reader, None)
            cluster_list = []
            for row in csv_reader:
                cluster_list.append(row[1])
        major_cluster = find_majority(cluster_list)[0]

        """5-0-4. Preparing train data with major_cluster
        """
        fe = FeatureExtractor()
        fe.raw_to_df("csv/tmp_joined.csv")
        print(major_cluster)
        fe.df_update(filter_by_membership, "NewCluster", major_cluster)
        fe.df_update(functions.selection, "playerid")
        fe.dump_df("csv/tmp_major_ids.csv")

        #Filtering test data by playerid(excerpted from cluser information) and age
        with open("csv/tmp_major_ids.csv") as f:
            csv_reader = reader(f)
            next(csv_reader, None)
            id_list = []
            for row in csv_reader:
                print(type(row[0]))
                id_list.append(int(float(row[0])))
        fe = FeatureExtractor()
        fe.raw_to_df(train_file_name)
        fe.df_update(filter_by_membership, "playerid", id_list)
        fe.df_update(filter_by_membership, "Age", age_high)
        fe.dump_df("csv/tmp_train.csv")
        
        """5-0-5. Preparing test data
        """
        fe = FeatureExtractor()
        fe.raw_to_df(test_base_file_name)
        fe.df_update(filter_by_membership, "playerid", test_id)
        fe.dump_df("csv/tmp_test.csv")

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
        xgbm.train("csv/tmp_train.csv", param_map, 100, randint(0,100))    
        xgbm.test("csv/tmp_test.csv", param_map)   
        xgbm.dump_output("output/predictor_using_kml/second_test.csv" , mode="a", header=False)    
    
    
    #6. Report Result
    """6-1. Raw Result"""

    """6-2. Distribution Similarity """

    """6-3. Reference: ZiPs """
    #End of Iteration

test_file.close()	

