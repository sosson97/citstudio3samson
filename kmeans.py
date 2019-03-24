from sklearn.cluster import KMeans
import numpy as np
from numpy import genfromtxt
import env
from models import Clusterer

input_path = env.train_input_name 
feature_start_index = env.feature_start_index
features_num = env.features_num

param_map = {
    "feature_start_index":env.feature_start_index,
    "features_num":env.features_num,
    "clusters_num":3
    }
                    

km = Clusterer()
km.kmeans("raw/last1year.csv", param_map, "output/clustered")

