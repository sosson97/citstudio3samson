#cluster_experiment.py

from feature_extractor import FeatureExtractor, OutputType
from functions import WAR2014to2016, join_with_2017, clustering, test_2017_train_less2017_split, cluster_split
from models import NN, XGBoostModel, SVRModel
from trainer import Trainer
from tester import Tester

import time
import env


if __name__ == "__main__":
    
    #1. create clustered data
    fe = FeatureExtractor()
    fe.raw_to_df("raw/last1year.csv")
   
    cluster_num = 5

    fe.df_update(clustering, cluster_num)
    path_list = []
    for i in range(cluster_num):
        path_list.append("raw/clustered_last1year" + str(i) + "of" + str(cluster_num) + ".csv")
    fe.dump_df(None, True, cluster_split, path_list, cluster_num)
   
    #2. for each, split it into train, test data
    for i in range(cluster_num):
        spfe = FeatureExtractor()
        spfe.raw_to_df("raw/clustered_last1year" + str(i)  + "of" + str(cluster_num) + ".csv")
        train_test = ["train_input/clustered_last1year_train" +str(i) + "of" + str(cluster_num) + ".csv", "test_input/clustered_last1year_test" +str(i) + "of" + str(cluster_num) +".csv"]
        spfe.dump_df(None, True, test_2017_train_less2017_split, train_test)


        #3. creating model
        #4. training    
        seed_list = [42]
        for seed in seed_list:
            xgbm = XGBoostModel()
            param_map = {
                "feature_start_index":env.feature_start_index,
                "features_num":env.features_num,
                "metric":"rmse"
                }

            xgbm.train(train_test[0], param_map, 1000, seed)
            xgbm.test(train_test[1], param_map)   
            xgbm.dump_output("output", "clustered_lasy1year" + str(i) + "of" + str(cluster_num) +"_output_" + str(seed) + ".csv")    
            print("xgb test dumped out for seed " + str(seed) + " and cluster" + str(i) + ".")    
