#models.py

#Abstract Implementation
from abc import ABC, abstractmethod

import torch.nn as nn
import xgboost as xgb
import env


#class: NN
class NN(nn.Module):
	#Internal		
	def __init__(self, parameters):
		super(NN, self).__init__()
		self.init_model_(parameters)
		self.net = nn.Sequential(
			nn.Linear(env.features_num, 512),	
			nn.LeakyReLU(),
			nn.Linear(512,50),
			nn.Dropout(0.9),
			nn.LeakyReLU(),
			nn.Linear(50,5),
			nn.LeakyReLU(),
			nn.Linear(5,1)
    )	
	
	def init_model_(self, parameters):
		print("NN model initiated")

	def forward(self, x):
		#print("forwarding CNN model...")
		return self.net(x)


#class: LSTM
#abstract class
class LSTM(ABC):
	#Internal
	def __init__(self, parameters):
		super.__init__()
		init_model_(parameters)

	def init_model_(parameters):
		print("LSTM model initiated")

	#API
	@abstractmethod
	def forward():
		print("forwarding LSTM model...")

#class: XGBoostModel
#Note: rain, test API should work without knowing schema of input data as long as both have same format and label at last column.
class XGBoostModel():
	#Internal
	#self.output contatins the result line by line
	def __init__(self, parameters):
		#super.__init__()
		self.model = None
		self.output = []
		self.init_model_(parameters)
	
	def init_model_(self, parameters):
		print("XGBoost model initiated")
	
	#API
	def train(self, train_parameter, num_round):
		print("XGBoost training starts")
		import numpy as np
		from numpy import genfromtxt
		train_data_np = genfromtxt(env.train_input_name, delimiter=',')	
		train_feature_np = np.array([l[env.feature_start_index:env.feature_start_index + env.features_num] for l in train_data_np][1:])
		train_label_np = np.array([l[env.feature_start_index + env.features_num] for l in train_data_np][1:])
		
		dtrain = xgb.DMatrix(train_feature_np, label=train_label_np)

		#dtest = xgb.DMatrix(test_feature, label=ytrue)

		num_round = 3000
		evallist = [(dtrain, 'train')]
		param = {'objective':'reg:linear',
							'eval_metric':'rmse',
							'learning_rate':0.01,
							'max_depth':15,
							'min_child_weight':1,
							'subsample':0.8,
							'colsample_bytree':0.6,
       				'gamma':0.5,
        			'reg_alpha':0.5,
        			'reg_lambda':1,
        			'seed':42}
		self.model = xgb.train(param, dtrain, num_round,evals=evallist)
		
		print(self.model.get_score(importance_type='gain'))
		print(self.model.get_score(importance_type='weight'))
	def test(self, test_parameters, path_test_input):
		print("XGBoost testing starts")
		import numpy as np
		from numpy import genfromtxt
		test_data_np = genfromtxt(env.test_input_name, delimiter=',')	
		test_feature_np = np.array([l[env.feature_start_index:env.feature_start_index + env.features_num] for l in test_data_np][1:])	
		test_label_np = np.array([l[env.feature_start_index + env.features_num] for l in test_data_np][1:])	
		dtest = xgb.DMatrix(test_feature_np, label=test_label_np)


	
		true = np.array([float(l[-1]) for l in test_data_np][1:])
		pred = self.model.predict(dtest)
		
		self.result = np.array(list(zip(true,pred)))
		for tr, pr in self.result:
			print("%.2f" % tr + " " + "%.2f" % pr)
		absdiff = 0
		for line in self.result:
			if line[0]-line[1] > 0:
				absdiff += line[0]-line[1]
			else:
				absdiff += line[1]-line[0]
		absdiff = absdiff/len(self.result)
		print("average abs diff = " + str(absdiff))
		
	
	def dump_output(self, dirname_output):
		f = open(dirname_output + "/output.txt", "w")
		for line in self.result:
			f.write(line + '\n')
		f.close()

		print("XGBoost dumped testing reuslt in " + dirname_output + "/output.txt")


"""
class CNNDemo(CNN):
	def forward(self, x):
		avg = 0;
		for val in x:
			avg += val
		return avg/len(x)
"""


