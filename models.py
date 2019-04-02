#models.py

#Abstract Implementation
from abc import ABC, abstractmethod

import torch.nn as nn
import xgboost as xgb
import env
from logger import Logger
import numpy as np
from numpy import genfromtxt

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
        train_data_np = genfromtxt(input_path, delimiter=',')   
        train_feature_np = np.array([l[train_parameter["feature_start_index"]:train_parameter["feature_start_index"] + train_parameter["features_num"]] for l in train_data_np][1:])
        train_label_np = np.array([l[train_parameter["feature_start_index"] + train_parameter["features_num"]] for l in train_data_np][1:])
        
        dtrain = xgb.DMatrix(train_feature_np, label=train_label_np)


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
        
        test_data_np = genfromtxt(input_path, delimiter=',')    
        test_feature_np = np.array([l[test_parameters["feature_start_index"]:test_parameters["feature_start_index"] + test_parameters["features_num"]] for l in test_data_np][1:])  
        test_label_np = np.array([l[test_parameters["feature_start_index"] + test_parameters["features_num"]] for l in test_data_np][1:])   
        dtest = xgb.DMatrix(test_feature_np, label=test_label_np)
        
        pid = np.array([int(l[1]) for l in test_data_np[1:]])
        true = np.array([float(l[test_parameters["feature_start_index"] + test_parameters["features_num"]]) for l in test_data_np][1:])
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
        if num_bg2 == 0:
            absdiff_bg2 = -1
        else:
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
        f.write("playerid, trueWAR, predWAR \n")
        for p, tr, pr in self.result:
            f.write(str(int(p)) + ", ") 
            f.write("%.2f" % tr + ", " + "%.2f" % pr + "\n")
        
        f.close()

        print("XGBoost dumped testing reuslt in " + dirname_output + "/" + output_name)





from sklearn.svm import SVR
class SVRModel():
    """class: SVRModel
        
        Description: Support Vector Regression model. 
    """
    #Internal
    def __init__(self):
        """Function: __init__

            Description: 
                create an empty SVR model
                
            Args:
                None

            Attributes:
                model (sklearn SVR Model): SVR model that the class holds currently.
                result (list): Testing output. Used for dumping out the output.
                logger (Logger): a logger for SVRModel

            Returns:
                None

        """ 
        self.model = None
        self.result = []
        self.logger = Logger("svr")

    def _csv_to_nparr(self, input_path, feature_start_index, features_num, pid=False):
        """Function: _df_to_nparr
            
            Description: 
                read csv file and transfrom it to feature matrix, label matrix according to the given index.

            Args:
                input_path (str): Relative or absolute path to input csv file. Input file must have features on columns from
                                (feature_start_index-th) columns to (feature_start_index-th + features_num - 1)-th
                                column, and have label at the next column of the last feature.
                feature_start_index (int): index where first feature resides.
                features_num (int): the number of featues.
                pid (bool): return numpy matrix of playerid if True, Default is False.

            Returns: 
                train_features_np (numpy matrix): sample_number * features_num, numpy matrix
                train_label_np (numpy matrix): sample_number * 1, numpy matrix
        """
        train_data_np = genfromtxt(input_path, delimiter=',')   
        
        
        train_feature_np = np.array([l[feature_start_index:feature_start_index + features_num] for l in train_data_np][1:])
        train_label_np = np.array([l[feature_start_index + features_num] for l in train_data_np][1:])
        if pid: 
            pid_np = np.array([l[feature_start_index-1] for l in train_data_np][1:]) 
            return pid_np, train_feature_np, train_label_np

        else:
            return train_feature_np, train_label_np 

    #API
    def train(self, input_path, train_parameters):
        """Function: train

            Description:
                Train SVR model with data in given path.   

            Args:
                input_path (str): path to train input.
                train_parameters (dic): a parameter dictionary for training.
                                                                the dictionary must contain follow entries.
                                                                1. feature_start_index
                                                              2. features_num

            Returns:
                None(self.model contains trained model after execution of this function though)

        """
        train_feature_np, train_label_np = self._csv_to_nparr(input_path, train_parameters["feature_start_index"],
                                                            train_parameters["features_num"])
        self.model = SVR(kernel='poly', gamma="scale", C=1.0, epsilon=0.2)
        self.model.fit(train_feature_np, train_label_np) #,sample_weight=train_label_np)
        self.logger.log("SVR trained with " + input_path)

    def test(self, input_path, test_parameters):
        """Function: test

            Description:
                Test SVR model with data in given path.   

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
        pid_np, test_feature_np, test_label_np = self._csv_to_nparr(input_path, test_parameters["feature_start_index"],
                                                            test_parameters["features_num"], True)
        
        
        pid = pid_np
        true = test_label_np
        pred = self.model.predict(test_feature_np)
 
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
        
        if num_bg2 == 0:
            absdiff_bg2 = -1    
        else:
            absdiff_bg2 = absdiff_bg2/num_bg2
        
    
        #for instant check.
        print(str(len(self.result)))
        print(str(num_bg2))
        print("average abs diff = " + str(absdiff))
        print("average abs diff in >2 = " + str(absdiff_bg2))  
         
        #logging the result.
        self.logger.log("SVR tested with " + input_path, True)     
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
        f.write("playerid, trueWAR, predWAR \n")
        for p, tr, pr in self.result:
            f.write(str(int(p)) + ", ") 
            f.write("%.2f" % tr + ", " + "%.2f" % pr + "\n")
        
        f.close()

        print("XGBoost dumped testing reuslt in " + dirname_output + "/" + output_name)




from sklearn.cluster import KMeans
class __Clusterer():
    """class: Clusterer
        Warining: This class is not actively used!! Look at cluster function in functions.py to use clustering. 
        
        Description: Provides several clustering algorithms 
    """
    #Internal
    def __init__(self):
        """Function: __init__

            Description: 
                create an empty Clusterer class. This class just provides some algorithms so there is no attribute or
                argument needed.
                
            Args:
                None

            Attributes:
                None

            Returns:
                None

        """ 

    def _csv_to_nparr(self, input_path, feature_start_index, features_num, pid=False):
        """Function: _df_to_nparr
            
            Description: 
                read csv file and transfrom it to feature matrix, label matrix according to the given index.

            Args:
                input_path (str): Relative or absolute path to input csv file. Input file must have features on columns from
                                (feature_start_index-th) columns to (feature_start_index-th + features_num - 1)-th
                                column, and have label at the next column of the last feature.
                feature_start_index (int): index where first feature resides.
                features_num (int): the number of featues.
                pid (bool): return numpy matrix of playerid if True, Default is False.

            Returns: 
                train_features_np (numpy matrix): sample_number * features_num, numpy matrix
                train_label_np (numpy matrix): sample_number * 1, numpy matrix
        """
        train_data_np = genfromtxt(input_path, delimiter=',')   
        
        
        train_feature_np = np.array([l[feature_start_index:feature_start_index + features_num] for l in train_data_np][1:])
        train_label_np = np.array([l[feature_start_index + features_num] for l in train_data_np][1:])
        if pid: 
            pid_np = np.array([l[feature_start_index-1] for l in train_data_np][1:]) 
            return pid_np, train_feature_np, train_label_np

        else:
            return train_feature_np, train_label_np 

    #API
    def kmeans(self, input_path, train_parameters, output_path):
        """Function: kmeans

            Description:
                Clustering given input file using K-means algorithm. 

            Args:
                input_path (str): path to train input.
                train_parameters (dic): a parameter dictionary for training.
                                                                the dictionary must contain follow entries.
                                                                1. feature_start_index
                                                                2. features_num
                                                                3. clusters_num

            Returns:
                None but clustered output is dumped though.

        """
        feature_np, label_np = self._csv_to_nparr(input_path, train_parameters["feature_start_index"],
                                                            train_parameters["features_num"])
        km = KMeans(n_clusters=train_parameters["clusters_num"], random_state=0).fit(feature_np)
        result = np.array(list(zip(feature_np,label_np)))
        
        file_list = []
        for i in range(train_parameters["clusters_num"]):
            file_list.append(open(output_path+str(i), "a+"))
        
        i = 0;
        for features, label in result:
            for entry in features:
                file_list[km.labels_[i]].write("%f" % entry + ", ")
            file_list[km.labels_[i]].write("%f" % label + "\n")
            i += 1
            
        for afile in file_list:
            afile.close()
        

