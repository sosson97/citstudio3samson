from feature_extractor import FeatureExtractor
from functions import rescaling, test_2017_train_less2017_split, clustering,cluster_split

fe = FeatureExtractor()
fe.raw_to_df("raw/last1year.csv")

fe.df_update(clustering, 3)
path_list = []
for i in range(3):
    path_list.append("raw/clustered_last1year" + str(i) + ".csv")
fe.dump_df(None, True, cluster_split, path_list, 3)
