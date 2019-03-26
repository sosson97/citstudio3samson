from feature_extractor import FeatureExtractor
from functions import rescaling, test_2017_train_less2017_split, clustering,cluster_split,null_remover
fe = FeatureExtractor()
fe.raw_to_df("raw/Fangraphs_1980-2017_raw.csv")
fe.df_update(null_remover)
#fe.dump_df("nullrm.csv")

path_list = ["train_input/train_simple_1year.csv", "test_input/test_simple_1year.csv"]
fe.dump_df(None, True, test_2017_train_less2017_split, path_list)


"""
fe = FeatureExtractor()
fe.raw_to_df("raw/last1year.csv")

fe.df_update(clustering, 3)
path_list = []
for i in range(3):
    path_list.append("raw/clustered_last1year" + str(i) + ".csv")
fe.dump_df(None, True, cluster_split, path_list, 3)
"""
