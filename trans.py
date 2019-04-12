from feature_extractor import FeatureExtractor
import functions

fe = FeatureExtractor()

col = ["Name", "playerid"]
for i in range(1,16):
    fe.raw_to_df("raw/1960-2018_WAR_enumerated.csv")
    fe.df_update(functions.null_remover, ["WAR"+str(i)])
    col.append("WAR"+str(i))
    fe.df_update(functions.selection,col)
    fe.dump_df("raw/players_until_career_" + str(i) + ".csv")


#fe.raw_to_df("raw/1960-2018.csv")
#fe.df_update(WAR_enumeration)
#fe.dump_df("raw/1960-2018_WAR_enumerated.csv")
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
