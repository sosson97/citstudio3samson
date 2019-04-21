import subprocess
from feature_extractor import FeatureExtractor
import functions
from models import XGBoostModel, SVRModel

import time
import env




#Note: Pseudo Code
test_players = read test_data.csv
for player in test_players:
    age = player[age] 
    fe = FeatureExtractor()
    fe.raw_to_df("raw/1960-2018_WAR_enumerated_by_age.csv")
    fe.update_df(WAR_enumerated_by_age)


