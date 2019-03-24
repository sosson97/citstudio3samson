from feature_extractor import FeatureExtractor
from functions import rescaling, test_2017_train_less2017_split

fe = FeatureExtractor()
fe.raw_to_df("raw/last1year.csv")
fe.df_update(rescaling)
fe.dump_df("train_input/scaled.csv", True, test_2017_train_less2017_split, "test_input/scaled.csv")
