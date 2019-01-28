import torch.nn as nn
import torch

import numpy as np


def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight.data,std=0.015)

    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data,std=0.015)
        # m.weight.data.normal_(0, 0.015)
        # m.bias.data.normal_(0, 0.015)

#---------------------------------
# Stochastic Multilayer Perceptron 
#---------------------------------
class BaseModel():
    def __init__(self, Input_dim=1, Output_dim=1, Hidden_size=100, Candidate_size=100,print_interval=50, plot_interval=1, plot_=False):
        super(BaseModel, self).__init__()
        self.input_dim = Input_dim
        self.output_dim = Output_dim
        self.candidate_size = Candidate_size
        self.hidden_size = Hidden_size
        self.Lambdas = [0.5, 1, 5, 10, 30, 50, 100, 150, 200, 250]
        self.r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
        self.tolerance = 0.0001
        self.loss = 10
        self.weight_IH = initWeight(self.input_dim, 1)
        self.bias_IH = initBiases()
        self.weight_HO = None
        self.weight_candidates = None
        self.bias_candidates = None
        # self.ridge_alpha = Ridge_alpha
        # self.regressor = Ridge(alpha=self.ridge_alpha)

        self.Print_interval = print_interval
        self.Plot_interval = plot_interval
        self.plot_ = plot_