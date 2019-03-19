#trainer.py

import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import env

class Trainer():
    """Class: Trainer

        Description: 
            Neural net trainer. For given neural net model, this class trains that model with given data and 
            provides a function dumping out trained model.

    """

    #Internal
    def __init__(self, parameters, optimizer, loss_function, model):
        """Function: __init__
            
            Description:
                Initialize training parameter, optimizer, loss function, model that will be trained.    

            Args:
                parameters (dic): a dictionary containing training parameters. 
                optimizer (PyTorch Optimizer): PyTorch optimizer for back-propagation such as Adam, SGD
                loss_function (str): Type of a loss function for back-propagation. Options are "MSE", "L1"
                model (NN Model): Neural net model such as CNN, LSTM. See models.py to know which options are availabel. Note: XGBoost, Prophet model should not be given as input.

            Attributes:
                parameters (dic): a dictionary containing training parameters. 
                optimizer (PyTorch Optimizer): PyTorch optimizer for back-propagation such as Adam, SGD
                loss_function (str): Type of a loss function for back-propagation. Options are "MSE", "L1"
                model (NN Model): Neural net model such as CNN, LSTM. See models.py to know which options are availabel. Note: XGBoost, Prophet model should not be given as input.

            Returns: 
                None

        """
        self.parameters = parameters
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.1)
        self.loss_function = nn.MSELoss()
        self.loader_data = self.train_data_load_()

    def _train_data_load(self):
        """Function: _train_data_load

            Description: 
                load training data from given path and make it a form of Torch Tensor(it is a data structure like matrix).  

            Args:
                input_path (str): path to input data

            Returns:
                train_data (DataLoader): train data formatted as torch DataLoader
        """
        
        import numpy as np
        from numpy import genfromtxt
        train_data_np = genfromtxt(env.train_input_name, delimiter=',') 
        train_feature_np = np.array([l[env.feature_start_index:env.feature_start_index + env.features_num] for l in train_data_np][1:])
        train_label_np = np.array([l[env.feature_start_index + env.features_num] for l in train_data_np][1:])

        class training(Dataset):
            def __init__(self):
                self.len = len(train_feature_np)
                self.x_data = torch.Tensor(train_feature_np)
                self.y_data = torch.Tensor(train_label_np)

            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]
    
            def __len__(self):
                return self.len
    
        train = training()
        return DataLoader(train, batch_size=5, shuffle=False)   

    #API
    def train(self, dirname_input, num_epochs):
        """Function: train

            Description:
                train self.model with self.optimizer, self.loss_function, and data in self.loaded_data  

            Args:
                num_epochs (int): the number of training epochs

            Returns:
                None but self.model now stores trained-model. You can use this model for testing.
        """
        print(self.model.__class__.__name__ + " model training starts...")
        
        dtype = torch.FloatTensor
        costs = []
        for epoch in range(num_epochs):
            cost = 0.0
            for t, (x, y) in enumerate(self.loader_data):
                x_var = x.type(dtype)
                y_var = y.type(dtype)
                scores = self.model(x_var).view(-1)
                
                loss = self.loss_function(scores, y_var)
                cost += loss.data[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            costs.append(cost / (t + 1))
            if (epoch + 1) % (num_epochs / 10) == 0:
                print('Epoch = %d, loss = %.4f' % (epoch + 1, costs[-1]))
         
        print("training done!")

    def dump_model(self, dirname_output, model_name):
        """Function: dump_model

            Description:
                dump out self.model as binary file using pickle
            
            Args: 
                output_path (str): Relative or absolute path of output.

            Returns:
                None but the model is dumped out.

        """
        print("Dumped trained model in " + dirname_output)
        f = open(dirname_output + "/" + model_name, "wb")
        pickle.dump(self.model, f)

