import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from ._linear_regression import LinearRegression
#-----------------------------------------
class ElasticRegression(LinearRegression):
    def __init__(self, 
                 learning_rate = 0.5,
                 l1_penalty    = 0.001,
                 l2_penalty    = 0.001,
                 epochs        = 100, 
                 weights       = None, 
                 bias          = None, 
                 batch_size    = 1000, 
                 random_state  = 42):
        
        super().__init__(learning_rate = learning_rate,
                         epochs        = epochs, 
                         weights       = weights, 
                         bias          = bias, 
                         batch_size    = batch_size, 
                         random_state  = random_state)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
    #---------------------------------
    def loss(self,yhat, y):   
        l1_term = self.l1_penalty*np.sum(self.weights[1:])/y.size
        l2_term = (self.l2_penalty/2)*np.sum(np.square(self.weights[1:]))/y.size
        return np.square(yhat - y).mean() + l1_term + l2_term
                  
    
    #---------------------------------
    def update(self):    
        l2_term = self.l2_penalty*np.sum(self.weights[1:])
        return self.weights - self.lr*self.grad + self.l1_penalty + l2_term