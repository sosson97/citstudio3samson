import subprocess
from feature_extractor import FeatureExtractor
import functions
from models import XGBoostModel, SVRModel

import time
import env




#Note: Pseudo Code
test_players = read test_data.csv
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




-> 이렇게 완성하면 raw data만 주면 모든걸 한 번에 해주는 아주 좋은 코드가 완성.
