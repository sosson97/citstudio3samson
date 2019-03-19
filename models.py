#models.py

#Abstract Implementation
from abc import ABC, abstractmethod

import torch.nn as nn
import xgboost as xgb
import env
from logger import Logger

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

class XGBoostModel():
    """Class: XGBoostModel
    
    Description:
        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and 
        portable(excerpt from https://xgboost.readthedocs.io/en/latest/). This model provides functions generating XGBoost 
        model, training XGBoost model, and testing trained-XGBoost model.   
    """

    #Internal
    def __init__(self):
        """Function: __init__

            Description: 
                create an empty XGBoost model
                
            Args:
                None

            Attributes:
                model (XGBoost Model): XGBoost model that the class holds currently.
                output (list): Testing output. Used for dumping out the output.
                logger (Logger): a logger for XGBoostModel

            Returns:
                None

        """
        self.model = None
        self.output = []
        self.logger = Logger("xgb")

    
    #API
    def train(self, input_path, train_parameter, num_round, seed=42):
        """Function: train

            Description:
                Train XGBoost model with data in given path.   

            Args:
                input_path (str): path to train input.
                train_parameters (dic): a parameter dictionary for training.
                                                                the dictionary must contain follow entries.
                                                                1. feature_start_index
                                                              2. features_num
                                                                3. metric('rmse', 'mae')
                num_round (int): the number of training round.
                seed (int): random seed. Default is 42.

            Returns:
                None(self.model contains trained model after execution of this function though)

        """

        print("XGBoost training starts")
        import numpy as np
        from numpy import genfromtxt
        train_data_np = genfromtxt(input_path, delimiter=',')   
        train_feature_np = np.array([l[train_parameter["feature_start_index"]:train_parameter["feature_start_index"] + train_parameter["features_num"]] for l in train_data_np][1:])
        train_label_np = np.array([l[train_parameter["feature_start_index"] + train_parameter["features_num"]] for l in train_data_np][1:])
        
        dtrain = xgb.DMatrix(train_feature_np, label=train_label_np)


        num_round = 3000
        evallist = [(dtrain, 'train')]
        param = {'objective':'reg:linear',
                            'eval_metric': train_parameter["metric"],
                            'learning_rate':0.01,
                            'max_depth':15,
                            'min_child_weight':1,
                            'subsample':0.8,
                            'colsample_bytree':0.6,
                    'gamma':0.5,
                    'reg_alpha':0.5,
                    'reg_lambda':1,
                    'seed':seed}
        self.model = xgb.train(param, dtrain, num_round,evals=evallist)
        
        #print(self.model.get_score(importance_type='gain'))
        #print(self.model.get_score(importance_type='weight'))
        self.logger.log("XGB trained with " + input_path)
    
    def test(self, input_path, test_parameters):
        """Function: test

            Description:
                Test XGBoost model with data in given path.   

            Args:
                input_path (str): Path to test data.
                test_parameters (dic): A parameter dictionary for testing.
                                                             the dictionary must contain the following entries.
                                                                1. feature_start_index
                                                              2. features_num


            Returns:
                None(self.output contains output line by line after execution of this function though. 
                        Furthermore, you can dump out the result by calling dump_output)

        """


        print("XGBoost testing starts")
        
        import numpy as np
        from numpy import genfromtxt
        test_data_np = genfromtxt(input_path, delimiter=',')    
        test_feature_np = np.array([l[test_parameters["feature_start_index"]:test_parameters["feature_start_index"] + test_parameters["features_num"]] for l in test_data_np][1:])  
        test_label_np = np.array([l[test_parameters["feature_start_index"] + test_parameters["features_num"]] for l in test_data_np][1:])   
        dtest = xgb.DMatrix(test_feature_np, label=test_label_np)
        
        pid = np.array([int(l[1]) for l in test_data_np[1:]])
        true = np.array([float(l[-1]) for l in test_data_np][1:])
        pred = self.model.predict(dtest)
        
        self.result = np.array(list(zip(pid,true,pred)))

        for p, tr, pr in self.result:
            print(int(p), end=" ") 
            print("%.2f" % tr + " " + "%.2f" % pr)
        absdiff = 0
        absdiff_bg2 = 0
        num_bg2 = 0
        for line in self.result:
            if line[1]-line[2] > 0:
                absdiff += line[1]-line[2]
                if line[1] >= 2:
                    absdiff_bg2 += line[1]-line[2]
                    num_bg2 += 1
            else:
                absdiff += line[2]-line[1]
                if line[1] >= 2:
                    absdiff_bg2 += line[2]-line[1]
                    num_bg2 += 1
        absdiff = absdiff/len(self.result)
        absdiff_bg2 = absdiff_bg2/num_bg2
        
    
        #for instant check.
        print(str(len(self.result)))
        print(str(num_bg2))
        print("average abs diff = " + str(absdiff))
        print("average abs diff in >2 = " + str(absdiff_bg2))  
        
        
        #logging the result.
        self.logger.log("XGB tested with " + input_path, True)     
        self.logger.log("average abs diff = " + str(absdiff))
        self.logger.log("average abs diff in >2 = " + str(absdiff_bg2))
    
    def dump_output(self, dirname_output, output_name):
        """Function: dump_output

            Description:
                Dump out the result in self.result to the given path.

            Args:
                output_path (str): Relative or absolute path of output.
        
            Returns: 
                None(dumped file)


        """
        f = open(dirname_output + "/" + output_name, "w")
        for p, tr, pr in self.result:
            f.write(str(int(p)) + ", ") 
            f.write("%.2f" % tr + ", " + "%.2f" % pr + "\n")
        
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


